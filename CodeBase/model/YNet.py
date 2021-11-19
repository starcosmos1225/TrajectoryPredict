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

import time

class YNetEncoder(nn.Module):
    def __init__(self, in_channels, channels=(64, 128, 256, 512, 512)):
        """
        Encoder model
        :param in_channels: int, semantic_classes + obs_len
        :param channels: list, hidden layer channels
        """
        super(YNetEncoder, self).__init__()
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


class YNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, output_len, traj=False):
        """
        Decoder models
        :param encoder_channels: list, encoder channels, used for skip connections
        :param decoder_channels: list, decoder channels
        :param output_len: int, pred_len
        :param traj: False or int, if False -> Goal and waypoint predictor, if int -> number of waypoints
        """
        super(YNetDecoder, self).__init__()

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


class YNetTorch(nn.Module):
    def __init__(self, obs_len, pred_len, segmentation_model_fp, use_features_only=False, semantic_classes=6,
                             encoder_channels=[], decoder_channels=[], waypoints=1,size=0.25, device='cuda:0'):
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
        super(YNetTorch, self).__init__()

        if segmentation_model_fp is not None and use_features_only:
            semantic_classes = 16  # instead of classes use number of feature_dim

        self.encoder = YNetEncoder(in_channels=semantic_classes + obs_len, channels=encoder_channels)

        self.goal_decoder = YNetDecoder(encoder_channels, decoder_channels, output_len=pred_len)
        self.traj_decoder = YNetDecoder(encoder_channels, decoder_channels, output_len=pred_len, traj=len(waypoints))

        self.softargmax_ = SoftArgmax2D(normalized_coordinates=False)
        self.waypoints = waypoints
        

    # def segmentation(self, image):
    #     return self.semantic_segmentation(image)

    # Forward pass for goal decoder
    def forward(self, obs, otherInp=None, extraInfo=None, params=None):
        if self.training:
            observedMap, _, gtWaypointMap, semanticMap = otherInp
            _, _, H, W = semanticMap.shape
            # print("obsmap:{} semanticmap:{}".format(observedMap.shape,semanticMap.shape))
            featureInput = torch.cat([semanticMap, observedMap], dim=1)

            features = self.pred_features(featureInput)

            predGoalMap = self.pred_goal(features)
            gtWaypointsMapsDownsampled = [nn.AvgPool2d(kernel_size=2**i, stride=2**i)(gtWaypointMap) for i in range(1, len(features))]
            gtWaypointsMapsDownsampled = [gtWaypointMap] + gtWaypointsMapsDownsampled
                        
            trajInput = [torch.cat([feature, goal], dim=1) for feature, goal in zip(features, gtWaypointsMapsDownsampled)]
            predTrajMap = self.pred_traj(trajInput)

            pred = self.softargmax(predTrajMap)
            predGoal = self.softargmax(predGoalMap[:, -1:])
            return pred, (predTrajMap,predGoalMap,predGoal)
        else:
            # start = time.time()

            observedMap, _, gtWaypointMap, semanticMap = otherInp
            inputTemplate = extraInfo
            _, _, H, W = semanticMap.shape
            temperature = params.test.temperature

            # Forward pass
            # Calculate features
            featureInput = torch.cat([semanticMap, observedMap], dim=1)
            features = self.pred_features(featureInput)
            # for f in features:
            #     print("features:{}".format(f.dtype))
            # Predict goal and waypoint probability distributions
            predWaypointMap = self.pred_goal(features)
            predWaypointMap = predWaypointMap[:, self.waypoints]

            predWaypointMapSigmoid = predWaypointMap / temperature
            predWaypointMapSigmoid = self.sigmoid(predWaypointMapSigmoid)
            # print("pred goal time:{}".format(time.time()-start))
            # start=time.time()
            ################################################ TTST ##################################################
            use_TTST = params.test.use_TTST
            if use_TTST:
                # TTST Begin
                # sample a large amount of goals to be clustered
                rel_thresh = params.test.rel_threshold
                
                goalSamples = sampling(predWaypointMapSigmoid[:, -1:], num_samples=10000, replacement=True, rel_threshold=rel_thresh)
                goalSamples = goalSamples.permute(2, 0, 1, 3)
                # print("sampling goal time:{}".format(time.time()-start))
                # start=time.time()
                numClusters = params.dataset.num_goals - 1
                goalSamplesSoftargmax = self.softargmax(predWaypointMap[:, -1:])  # first sample is softargmax sample
                # Iterate through all person/batch_num, as this k-Means implementation doesn't support batched clustering
                goalSamplesList = []
                for person in range(goalSamples.shape[1]):
                    goalSample = goalSamples[:, person, 0]

                    # Actual k-means clustering, Outputs:
                    # cluster_ids_x -  Information to which cluster_idx each point belongs to
                    # cluster_centers - list of centroids, which are our new goal samples
                    clusterIdsX, clusterCenters = kmeans(X=goalSample, num_clusters=numClusters, distance='euclidean', device=params.device, tqdm_flag=False, tol=0.001, iter_limit=1000)
                    goalSamplesList.append(clusterCenters)
                # print("kmeans time:{} num:{}".format(time.time()-start, goalSamples.shape[1]))
                # start=time.time()

                goalSamples = torch.stack(goalSamplesList).permute(1, 0, 2).unsqueeze(2)
                goalSamples = torch.cat([goalSamplesSoftargmax.unsqueeze(0), goalSamples], dim=0)
                # TTST End

            # Not using TTST
            else:
                goalSamples = sampling(predWaypointMapSigmoid[:, -1:], num_samples=params.dataset.num_goals)
                goalSamples = goalSamples.permute(2, 0, 1, 3).contiguous()

            # Predict waypoints:
            # in case len(waypoints) == 1, so only goal is needed (goal counts as one waypoint in this implementation)
            if len(self.waypoints) == 1:
                # print("length=1:{} continuse:{}".format(goalSamples.device,goalSamples.is_contiguous()))
                waypointSamples = goalSamples
            # print("TTST time:{}".format(time.time()-start))
            # start=time.time()
            ################################################ CWS ###################################################
            # CWS Begin
            use_CWS = params.test.use_CWS
            if use_CWS and len(self.waypoints) > 1:
                sigmaFactor = params.test.CWS_params.sigma_factor
                ratio = params.test.CWS_params.ratio
                rot = params.test.CWS_params.rot

                goalSamples = goalSamples.repeat(params.dataset.num_traj, 1, 1, 1)  # repeat K_a times
                lastObserved = obs[:,-1,:].to(params.device)  # [N, 2]
                waypointSamplesList = []  # in the end this should be a list of [K, N, # waypoints, 2] waypoint coordinates
                for gNum, waypointSamples in enumerate(goalSamples.squeeze(2)):
                    waypointList = []  # for each K sample have a separate list
                    waypointList.append(waypointSamples)

                    for waypointNum in reversed(range(len(self.waypoints)-1)):
                        distance = lastObserved - waypointSamples
                        gaussianHeatmaps = []
                        trajIdx = gNum // params.dataset.num_goals  # idx of trajectory for the same goal
                        for dist, coordinate in zip(distance, waypointSamples):  # for each person
                            lengthRatio = 1 / (waypointNum + 2)
                            gaussMean = coordinate + (dist * lengthRatio)  # Get the intermediate point's location using CV model
                            sigmaFactor_ = sigmaFactor - trajIdx
                            gaussianHeatmaps.append(torch_multivariate_gaussian_heatmap(gaussMean, H, W, dist, sigmaFactor_, ratio, params.dataset.device, rot))
                        gaussianHeatmaps = torch.stack(gaussianHeatmaps)  # [N, H, W]

                        waypointMapBefore = predWaypointMapSigmoid[:, waypointNum]
                        waypointMap = waypointMapBefore * gaussianHeatmaps
                        # normalize waypoint map
                        waypointMap = (waypointMap.flatten(1) / waypointMap.flatten(1).sum(-1, keepdim=True)).view_as(waypointMap)

                        # For first traj samples use softargmax
                        if gNum // params.dataset.num_goals == 0:
                            # Softargmax
                            waypointSamples = self.softargmax_on_softmax_map(waypointMap.unsqueeze(0))
                            waypointSamples = waypointSamples.squeeze(0)
                        else:
                            waypointSamples = sampling(waypointMap.unsqueeze(1), num_samples=1, rel_threshold=0.05)
                            waypointSamples = waypointSamples.permute(2, 0, 1, 3)
                            waypointSamples = waypointSamples.squeeze(2).squeeze(0)
                        waypointList.append(waypointSamples)

                    waypointList = waypointList[::-1]
                    waypointList = torch.stack(waypointList).permute(1, 0, 2)  # permute back to [N, # waypoints, 2]
                    waypointSamplesList.append(waypointList)
                waypointSamples = torch.stack(waypointSamplesList)

                # CWS End

            # If not using CWS, and we still need to sample waypoints (i.e., not only goal is needed)
            elif not use_CWS and len(self.waypoints) > 1:
                waypointSamples = sampling(predWaypointMapSigmoid[:, :-1], num_samples=params.dataset.num_goals * params.dataset.num_traj)
                waypointSamples = waypointSamples.permute(2, 0, 1, 3)
                goalSamples = goalSamples.repeat(params.dataset.num_traj, 1, 1, 1)  # repeat K_a times
                waypointSamples = torch.cat([waypointSamples, goalSamples], dim=2)

            # Interpolate trajectories given goal and waypoints
            # print("CWS time:{}".format(time.time()-start))
            # start=time.time()
            futureSamples = []
            
            # print("input template time:{}".format(time.time()-start))
            # start=time.time()
            # print("inputtmeplate:{}".format(inputTemplate.dtype))
            waypoints = waypointSamples.cpu()
            for waypoint in waypoints:
                waypointMap = getPatch(inputTemplate, waypoint.reshape(-1, 2).numpy(), H, W)
                # print("getpatch time:{}".format(time.time()-start))
                # start=time.time()
                # for w in waypointMap:
                #     print("way:{}".format(w.dtype))
                waypointMap = torch.stack(waypointMap).reshape([-1, len(self.waypoints), H, W])

                waypointMapsDownsampled = [nn.AvgPool2d(kernel_size=2 ** i, stride=2 ** i)(waypointMap) for i in range(1, len(features))]
                waypointMapsDownsampled = [waypointMap] + waypointMapsDownsampled
                # for w in waypointMapsDownsampled:
                #     print("waypoint :{}".format(w.dtype))
                trajInput = [torch.cat([feature, goal], dim=1) for feature, goal in zip(features, waypointMapsDownsampled)]
                # print("downsampled time:{}".format(time.time()-start))
                # start=time.time()
                # for t in trajInput:
                #     print("traj:{}".format(t.dtype))
                predTrajMap = self.pred_traj(trajInput)
                # print("pred time:{}".format(time.time()-start))
                # start=time.time()
                predTraj = self.softargmax(predTrajMap)
                futureSamples.append(predTraj)
            # print("after TTST time:{}".format(time.time()-start))
            # start=time.time()
            futureSamples = torch.stack(futureSamples)
            return futureSamples, (waypointSamples)



    def pred_goal(self, features):
        goal = self.goal_decoder(features)
        return goal

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

class YNetTorchNoGoal(nn.Module):
    def __init__(self, obs_len, pred_len, segmentation_model_fp, use_features_only=False, semantic_classes=6,
                             encoder_channels=[], decoder_channels=[], waypoints=1):
        super(YNetTorch, self).__init__()

        if segmentation_model_fp is not None and use_features_only:
            semantic_classes = 16  # instead of classes use number of feature_dim

        self.encoder = YNetEncoder(in_channels=semantic_classes + obs_len, channels=encoder_channels)

        # self.goal_decoder = YNetDecoder(encoder_channels, decoder_channels, output_len=pred_len)
        self.traj_decoder = YNetDecoder(encoder_channels, decoder_channels, output_len=pred_len, traj=len(waypoints))

        self.softargmax_ = SoftArgmax2D(normalized_coordinates=False)
        self.waypoints = waypoints
        

    # def segmentation(self, image):
    #     return self.semantic_segmentation(image)

    # Forward pass for goal decoder
    def forward(self, obs, otherInp=None, extraInfo=None, params=None):
        if self.training:
            observedMap, _, gtWaypointMap, semanticMap = otherInp
            _, _, H, W = semanticMap.shape
            featureInput = torch.cat([semanticMap, observedMap], dim=1)
            features = self.pred_features(featureInput)
            trajInput = features # [torch.cat([feature, goal], dim=1) for feature, goal in zip(features, gtWaypointsMapsDownsampled)]
            predTrajMap = self.pred_traj(trajInput)
            pred = self.softargmax(predTrajMap)
            # predGoal = self.softargmax(predGoalMap[:, -1:])
            return pred, (predTrajMap,None,None)
        else:
            # start = time.time()

            observedMap, _, gtWaypointMap, semanticMap = otherInp
            inputTemplate = extraInfo
            _, _, H, W = semanticMap.shape
            temperature = params.test.temperature

            # Forward pass
            # Calculate features
            featureInput = torch.cat([semanticMap, observedMap], dim=1)
            features = self.pred_features(featureInput)
        
            futureSamples = []
            
            # print("input template time:{}".format(time.time()-start))
            # start=time.time()
            # print("inputtmeplate:{}".format(inputTemplate.dtype))
    
            trajInput = features 
            predTrajMap = self.pred_traj(trajInput)
                # print("pred time:{}".format(time.time()-start))
                # start=time.time()
            for _ in range(params.dataset.num_Traj):
                predTraj = self.samplingTrajFrom(predTrajMap)
                futureSamples.append(predTraj)
            # print("after TTST time:{}".format(time.time()-start))
            # start=time.time()
            futureSamples = torch.stack(futureSamples)
            return futureSamples, None



    def pred_goal(self, features):
        goal = self.goal_decoder(features)
        return goal

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


                



























