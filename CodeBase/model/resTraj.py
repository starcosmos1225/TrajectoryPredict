
import torch
import torch.nn as nn
from .backbone.resnet import resnet18, resnet50, resnet101, resnet152, \
    resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2, \
        conv3x3

resnetDict = {
    'resnet18': resnet18,
    'resnet50': resnet50, 
    'resnet101': resnet101, 
    'resnet152': resnet152,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x8d':resnext101_32x8d, 
    'wide_resnet50_2': wide_resnet50_2, 
    'wide_resnet101_2': wide_resnet101_2
}


from tqdm import tqdm
import numpy as np

from utils.softargmax import SoftArgmax2D, create_meshgrid
from data.image_utils import getPatch
from utils.utils import sampling, torch_multivariate_gaussian_heatmap
from utils.kmeans import kmeans

import time

class resnetTraj(nn.Module):
    def __init__(self, obs_len, 
                       pred_len, 
                       segmentation_model_fp, 
                       use_features_only=False, 
                       semantic_classes=6, 
                       width=16,
                       resnet_name='resnet50'):
        """
        Complete Y-net Architecture including semantic segmentation backbone, heatmap embedding and ConvPredictor
        :param obs_len: int, observed timesteps
        :param pred_len: int, predicted timesteps
        :param segmentation_model_fp: str, filepath to pretrained segmentation model
        :param use_features_only: bool, if True -> use segmentation features from penultimate layer, if False -> use softmax class predictions
        :param semantic_classes: int, number of semantic classes
        :param encoder_channels: list, encoder channel structure
        :param decoder_channels: list, decoder channel structure
        :param num_waypoints: int, number of waypoints
        """
        super(resnetTraj, self).__init__()
        self.predLength = pred_len
        if segmentation_model_fp is not None and use_features_only:
            semantic_classes = 16  # instead of classes use number of feature_dim
        width_per_layer = []
        for _ in range(4):
            width_per_layer.append(width*pred_len)
            width*=2
        # print(width_per_layer)
        width = width //2
        self.baseModel = resnetDict[resnet_name](in_channels=semantic_classes + obs_len, 
                                                 width_per_layer=width_per_layer)

        self.decoder = nn.Sequential(
            #b, cin,h,w -> b, prelen*2, h,w
                conv3x3(in_planes=self.baseModel.inplanes, out_planes=pred_len*2, groups=pred_len),
                nn.Tanh(),
                nn.BatchNorm2d(pred_len*2),
                nn.AdaptiveAvgPool2d((1,1))
        )
        # self.activate = nn.Tanh()
        
    def forward(self, obs, otherInp=None, extraInfo=None, params=None):
        if self.training:
            observedMap, _, _, semanticMap = otherInp
            _, _, H, W = semanticMap.shape
            # print("obsmap:{} semanticmap:{}".format(observedMap.shape,semanticMap.shape))
            featureInput = torch.cat([semanticMap, observedMap], dim=1)
            # print("input shape:{}".format(featureInput.shape))
            features = self.baseModel(featureInput)
            # print("output features:{}".format(features.shape))
            pred = self.decoder(features)
            # pred = out.tanh()
            pred = pred.view(-1,self.predLength,2)
            pred[:,:,0] = (pred[:,:,0] + 1.0)*W*0.5
            pred[:,:,1] = (pred[:,:,1] + 1.0)*H*0.5
            return pred, None
        else:
            observedMap, _, _, semanticMap = otherInp
            _, _, H, W = semanticMap.shape
            featureInput = torch.cat([semanticMap, observedMap], dim=1)
            features = self.baseModel(featureInput)
            pred = self.decoder(features)
            # pred = out.tanh()
            pred = pred.view(-1,self.predLength,2)
            # x = (x+1.0)*W/2
            # y = (y+1.0)*H/2 
            pred[:,:,0] = (pred[:,:,0] + 1.0)*W*0.5
            pred[:,:,1] = (pred[:,:,1] + 1.0)*H*0.5
            return pred
    
    def load(self, path):
        print(self.load_state_dict(torch.load(path)))
    
    def save(self, path):
        torch.save(self.state_dict(), path)
