from json import encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam,SGD, RMSprop, Adagrad
from tqdm import tqdm
import numpy as np

from utils.softargmax import SoftArgmax2D, create_meshgrid
from data.image_utils import getPatch
from utils.utils import sampling, torch_multivariate_gaussian_heatmap
from utils.kmeans import kmeans
# from utils.utils import samplingTrajFromHeatMap

import time

class UNetEncoder(nn.Module):
    def __init__(self, in_channels, channels=(64, 128, 256, 512, 512)):
        """
        Encoder model
        :param in_channels: int, semantic_classes + obs_len
        :param channels: list, hidden layer channels
        """
        super(UNetEncoder, self).__init__()
        self.stages = nn.ModuleList()

        # First block
        self.stages.append(nn.Sequential(
                nn.Conv2d(in_channels, channels[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True),
        ))

        # Subsequent blocks, each starting with MaxPool
        for i in range(len(channels)-1):
            self.stages.append(nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                    nn.Conv2d(channels[i], channels[i+1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels[i+1], channels[i+1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.ReLU(inplace=True)))

        # Last MaxPool layer before passing the features into decoder
        self.stages.append(nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)))

    def forward(self, x):
        # Saves the feature maps Tensor of each layer into a list, as we will later need them again for the decoder
        features = []
        for stage in self.stages:
                x = stage(x)
                features.append(x)
        return features


class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, output_len, traj=False):
        """
        Decoder models
        :param encoder_channels: list, encoder channels, used for skip connections
        :param decoder_channels: list, decoder channels
        :param output_len: int, pred_len
        :param traj: False or int, if False -> Goal and waypoint predictor, if int -> number of waypoints
        """
        super(UNetDecoder, self).__init__()

        # The trajectory decoder takes in addition the conditioned goal and waypoints as an additional image channel
        if traj:
            encoder_channels = [channel+traj for channel in encoder_channels]
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder
        center_channels = encoder_channels[0]

        decoder_channels = decoder_channels

        # The center layer (the layer with the smallest feature map size)
        self.center = nn.Sequential(
                nn.Conv2d(center_channels, center_channels*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(center_channels*2, center_channels*2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True)
        )

        # Determine the upsample channel dimensions
        upsample_channels_in = [center_channels*2] + decoder_channels[:-1]
        upsample_channels_out = [num_channel // 2 for num_channel in upsample_channels_in]

        # Upsampling consists of bilinear upsampling + 3x3 Conv, here the 3x3 Conv is defined
        self.upsample_conv = [
                nn.Conv2d(in_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                for in_channels_, out_channels_ in zip(upsample_channels_in, upsample_channels_out)]
        self.upsample_conv = nn.ModuleList(self.upsample_conv)

        # Determine the input and output channel dimensions of each layer in the decoder
        # As we concat the encoded feature and decoded features we have to sum both dims
        in_channels = [enc + dec for enc, dec in zip(encoder_channels, upsample_channels_out)]
        out_channels = decoder_channels

        self.decoder = [nn.Sequential(
                nn.Conv2d(in_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels_, out_channels_, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.ReLU(inplace=True))
                for in_channels_, out_channels_ in zip(in_channels, out_channels)]
        self.decoder = nn.ModuleList(self.decoder)


        # Final 1x1 Conv prediction to get our heatmap logits (before softmax)
        self.predictor = nn.Conv2d(in_channels=decoder_channels[-1], out_channels=output_len, kernel_size=1, stride=1, padding=0)

    def forward(self, features):
        # Takes in the list of feature maps from the encoder. Trajectory predictor in addition the goal and waypoint heatmaps
        features = features[::-1]  # reverse the order of encoded features, as the decoder starts from the smallest image
        center_feature = features[0]
        x = self.center(center_feature)
        for i, (feature, module, upsample_conv) in enumerate(zip(features[1:], self.decoder, self.upsample_conv)):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # bilinear interpolation for upsampling
            x = upsample_conv(x)  # 3x3 conv for upsampling
            x = torch.cat([x, feature], dim=1)  # concat encoder and decoder features
            x = module(x)  # Conv
        x = self.predictor(x)  # last predictor layer
        return x


class UNet(nn.Module):
    def __init__(self, pred_len, segmentation_model_fp, use_features_only=False, semantic_classes=6,
                             encoder_channels=[], decoder_channels=[]):
        
        super(UNet, self).__init__()

        if segmentation_model_fp is not None and use_features_only:
            semantic_classes = 16  # instead of classes use number of feature_dim

        self.encoder = UNetEncoder(in_channels=semantic_classes + pred_len, channels=encoder_channels)

        self.traj_decoder = UNetDecoder(encoder_channels, decoder_channels, output_len=pred_len)
    
        self.softargmax_ = SoftArgmax2D(normalized_coordinates=False)
        # self.waypoints = waypoints
        

    # def segmentation(self, image):
    #     return self.semantic_segmentation(image)

    # Forward pass for goal decoder
    def forward(self, obs, otherInp=None, extraInfo=None, params=None):
       
        semanticMap,_,initTrajMap = otherInp
        # print(semanticMap.shape)
        # print(initTrajMap.shape)
        b, num_class, H, W = semanticMap.shape
        num_traj = params.dataset.num_traj
        semanticMap = semanticMap.unsqueeze(1).repeat(1,num_traj,1,1,1)
        semanticMap = semanticMap.view(b*num_traj,-1, H, W)
        initTrajMap = initTrajMap.view(b*num_traj,-1,H,W)
        # print(semanticMap.shape)
        # print(initTrajMap.shape)
        # print("obsmap:{} semanticmap:{}".format(observedMap.shape,semanticMap.shape))
        featureInput = torch.cat([semanticMap, initTrajMap], dim=1)
        # print(featureInput.shape)
        # t=input()
        features = self.pred_features(featureInput)

        predTrajMap = self.pred_traj(features)

        pred = self.softargmax(predTrajMap)
        # print(pred.shape)
        # print(predTrajMap.shape)
        # t=input()
        if self.training:
            return pred, (predTrajMap, None)
        else:
            return pred
        


    # def pred_goal(self, features):
    #     goal = self.goal_decoder(features)
    #     return goal

    # Forward pass for trajectory decoder
    def pred_traj(self, features):
        traj = self.traj_decoder(features)
        return traj

    # Forward pass for feature encoder, returns list of feature maps
    def pred_features(self, x):
        features = self.encoder(x)
        return features

    # Softmax for Image data as in dim=NxCxHxW, returns softmax image shape=NxCxHxW
    def softmax(self, x):
        return nn.Softmax(2)(x.view(*x.size()[:2], -1)).view_as(x)

    # Softargmax for Image data as in dim=NxCxHxW, returns 2D coordinates=Nx2
    def softargmax(self, output):
        return self.softargmax_(output)

    def sigmoid(self, output):
        return torch.sigmoid(output)

    def softargmax_on_softmax_map(self, x):
        """ Softargmax: As input a batched image where softmax is already performed (not logits) """
        pos_y, pos_x = create_meshgrid(x, normalized_coordinates=False)
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)
        x = x.flatten(2)

        estimated_x = pos_x * x
        estimated_x = torch.sum(estimated_x, dim=-1, keepdim=True)
        estimated_y = pos_y * x
        estimated_y = torch.sum(estimated_y, dim=-1, keepdim=True)
        softargmax_coords = torch.cat([estimated_x, estimated_y], dim=-1)
        return softargmax_coords
    
    def load(self, path):
        print(self.load_state_dict(torch.load(path)))
    
    def save(self, path):
        torch.save(self.state_dict(), path)

