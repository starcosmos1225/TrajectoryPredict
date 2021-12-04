
import torch
import torch.nn as nn
from .backbone.resnet import resnet18, resnet50, resnet101, resnet152, \
    resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2, \
        conv3x3
from .backbone.Linear import MLP,LinearEmbedding


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
            return pred,None
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


class CVAEresnetTraj(nn.Module):
    def __init__(self, obs_len, 
                       pred_len, 
                       z_dim,
                       sigma,
                       lastlayer_dim,
                       segmentation_model_fp, 
                       use_features_only=False, 
                       semantic_classes=6, 
                       resnet_name='resnet50'):
        
        super(CVAEresnetTraj, self).__init__()
        self.predLength = pred_len
        self.sigma = sigma
        if segmentation_model_fp is not None and use_features_only:
            semantic_classes = 16  # instead of classes use number of feature_dim
        width_per_layer = [64,128,256,lastlayer_dim]
        self.obsEncoder = resnetDict[resnet_name](in_channels=semantic_classes + obs_len,
                                                width_per_layer=width_per_layer)
        self.predEncoder = resnetDict[resnet_name](in_channels=semantic_classes + pred_len,
                                                width_per_layer=width_per_layer) 
        
        traj_dim = self.obsEncoder.inplanes
        self.cvaeEncoder = MLP(traj_dim*2,2*z_dim,z_dim)
        self.decoder = MLP(traj_dim+z_dim, pred_len*2, z_dim,activation='tanh')
        self.z_dim = z_dim
        # self.activate = nn.Tanh()
        
    def forward(self, obs, otherInp=None, extraInfo=None, params=None):
        device = params.device
        if self.training:
            observedMap, gtFutureMap, _, semanticMap = otherInp
            _, _, H, W = semanticMap.shape
            obsInput = torch.cat([semanticMap, observedMap], dim=1)
            obsFeatures = self.obsEncoder.getFeat(obsInput)
            predInput = torch.cat([semanticMap, gtFutureMap],dim=1)
            predFeatures = self.predEncoder.getFeat(predInput)
            latent = torch.cat([obsFeatures,predFeatures],dim=-1)
            latent = self.cvaeEncoder(latent)
            
            mu = latent[:, 0:self.z_dim] # 2-d array
            logvar = latent[:, self.z_dim:] # 2-d array

            var = logvar.mul(0.5).exp_()
            eps = torch.FloatTensor(var.size()).normal_()
            eps = eps.to(device)
            z = eps.mul(var).add_(mu)
            z = z.float().to(device)
            decoder_input = torch.cat((obsFeatures, z), dim = -1)
            pred = self.decoder(decoder_input)
            # pred = out.tanh()
            pred = pred.view(-1,self.predLength,2)
            pred[:,:,0] = (pred[:,:,0] + 1.0)*W*0.5
            pred[:,:,1] = (pred[:,:,1] + 1.0)*H*0.5
            return pred, {
                    'mean': mu, 
                    'var': logvar
                    }
        else:
            observedMap, gtFutureMap, _, semanticMap = otherInp
            _, _, H, W = semanticMap.shape
            obsInput = torch.cat([semanticMap, observedMap], dim=1)
            obsFeatures = self.obsEncoder.getFeat(obsInput)
            
            z = torch.Tensor(params.dataset.num_traj, obs.size(0), self.z_dim)
            z.normal_(0, self.sigma)
            z = z.float().to(device)

            obsFeatures = obsFeatures.unsqueeze(0).repeat(params.dataset.num_traj, 1, 1)
            decoder_input = torch.cat((obsFeatures, z), dim = -1)
            pred = self.decoder(decoder_input)
            # pred = out.tanh()
            pred = pred.view(params.dataset.num_traj, -1,self.predLength,2)
            pred[:,:,:,0] = (pred[:,:,:,0] + 1.0)*W*0.5
            pred[:,:,:,1] = (pred[:,:,:,1] + 1.0)*H*0.5
            return pred
    
    def load(self, path):
        print(self.load_state_dict(torch.load(path)))
    
    def save(self, path):
        torch.save(self.state_dict(), path)

class CVAETeacherTraj(nn.Module):
    def __init__(self, obs_len, 
                       pred_len, 
                       z_dim,
                       sigma,
                       feat_dim,
                       segmentation_model_fp, 
                       use_features_only=False, 
                       semantic_classes=6, 
                       resnet_name='resnet50'):
        
        super(CVAETeacherTraj, self).__init__()
        self.predLength = pred_len
        self.sigma = sigma
        if segmentation_model_fp is not None and use_features_only:
            semantic_classes = 16  # instead of classes use number of feature_dim
        width_per_layer = [64,128,256,512]
        self.obsEncoder = resnetDict[resnet_name](in_channels=semantic_classes + obs_len,
                                                width_per_layer=width_per_layer,num_classes=feat_dim)
        self.predEncoder = resnetDict[resnet_name](in_channels=semantic_classes + pred_len,
                                                width_per_layer=width_per_layer,num_classes=feat_dim, another_classes=z_dim) 
        
        self.cvaeEncoder = MLP(feat_dim+z_dim,2*z_dim,z_dim)
        self.studentDecoder = MLP(feat_dim+z_dim, feat_dim, (feat_dim+z_dim)//2)
        self.teacherDecoder = MLP(feat_dim, pred_len*2, z_dim,activation='tanh')
        self.z_dim = z_dim
        # self.activate = nn.Tanh()
        
    def forward(self, obs, otherInp=None, extraInfo=None, params=None):
        device = params.device
        if self.training:
            observedMap, gtFutureMap, _, semanticMap = otherInp
            _, _, H, W = semanticMap.shape
            obsInput = torch.cat([semanticMap, observedMap], dim=1)
            obsFeatures = self.obsEncoder(obsInput)
            predInput = torch.cat([semanticMap, gtFutureMap],dim=1)
            predFeatures, noiseFeatures = self.predEncoder(predInput)
            latent = torch.cat([obsFeatures,noiseFeatures],dim=-1)
            latent = self.cvaeEncoder(latent)
            
            mu = latent[:, 0:self.z_dim] # 2-d array
            logvar = latent[:, self.z_dim:] # 2-d array

            var = logvar.mul(0.5).exp_()
            eps = torch.FloatTensor(var.size()).normal_()
            eps = eps.to(device)
            z = eps.mul(var).add_(mu)
            z = z.float().to(device)
            decoder_input = torch.cat((obsFeatures, z), dim = -1)
            studentFeatures = self.studentDecoder(decoder_input)
            pred = self.teacherDecoder(predFeatures)
            # pred = out.tanh()
            pred = pred.view(-1,self.predLength,2)
            pred[:,:,0] = (pred[:,:,0] + 1.0)*W*0.5
            pred[:,:,1] = (pred[:,:,1] + 1.0)*H*0.5
            return pred, {
                    'mean': mu, 
                    'var': logvar,
                    'student': studentFeatures,
                    'teacher': predFeatures.detach(),
                    }
        else:
            observedMap, gtFutureMap, _, semanticMap = otherInp
            _, _, H, W = semanticMap.shape
            obsInput = torch.cat([semanticMap, observedMap], dim=1)
            obsFeatures = self.obsEncoder(obsInput)
            # print(obsFeatures.shape)
            z = torch.Tensor(params.dataset.num_traj, obs.size(0), self.z_dim)
            z.normal_(0, self.sigma)
            z = z.float().to(device)
            obsFeatures = obsFeatures.unsqueeze(0).repeat(params.dataset.num_traj, 1,1)
            decoder_input = torch.cat((obsFeatures, z), dim = -1)
            # print(z.shape)
            # print(decoder_input.shape)
            
            studentFeatures = self.studentDecoder(decoder_input)
            pred = self.teacherDecoder(studentFeatures)
            # pred = out.tanh()
            pred = pred.view(params.dataset.num_traj,-1,self.predLength,2)
            pred[:,:,:,0] = (pred[:,:,:,0] + 1.0)*W*0.5
            pred[:,:,:,1] = (pred[:,:,:,1] + 1.0)*H*0.5
            return pred
    
    def load(self, path):
        print(self.load_state_dict(torch.load(path)))
    
    def save(self, path):
        torch.save(self.state_dict(), path)


class CVAEResMLP(nn.Module):
    def __init__(self, obs_len,pred_len,semantic_classes, inp_size, out_size,z_size,sigma,
                   hidden_size=512,
                   resnet_name='resnet50'):
        super(CVAEResMLP, self).__init__()
        self.sigma = sigma

        self.obsEncoder = resnetDict[resnet_name](in_channels=semantic_classes + obs_len,
                                                num_classes=hidden_size)
        self.predEncoder = nn.Sequential(
                       LinearEmbedding(inp_size*pred_len, hidden_size),
                       MLP(hidden_size, hidden_size, hidden_size//2)
        )
        self.decoder = LinearEmbedding(hidden_size + z_size, out_size)
        self.noiseEncoder = MLP(hidden_size*2 , z_size * 2 , z_size)

        self.trajectoryDecoder = nn.Sequential(
                        MLP(hidden_size + z_size, hidden_size+z_size, (hidden_size+z_size)//2),
                        # LinearEmbedding(hidden_size+z_size,out_size))
                        LinearEmbedding(hidden_size+z_size, 2*pred_len))
    

        # self.numLayers = num_layers

        self.hiddenSize = hidden_size
        self.zSize = z_size
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, obs, otherInp=None,extraInfo=None,params=None):
        
        device = params.device
        observedMap, gtFutureMap, _, semanticMap,gt = otherInp
        _, _, H, W = semanticMap.shape
        obsInput = torch.cat([semanticMap, observedMap], dim=1)
        obsFeat = self.obsEncoder(obsInput)
        predLength = params.dataset.pred_len
        # print(obsFeat.shape)
        # pred[:,:,0] = (pred[:,:,0] + 1.0)*W*0.5
        # pred[:,:,1] = (pred[:,:,1] + 1.0)*H*0.5
        gt[:,:,0] = gt[:,:,0] * 2/W-1.0
        gt[:,:,1] = gt[:,:,1] * 2/H-1.0
        predFeat = self.predEncoder(gt.view(-1,predLength*2))
        # print(predFeat.shape)
        # t=input()
        if self.training:
            # create z noise
            # (batch 2* hiddensize)
            noiseInp = torch.cat((obsFeat, predFeat),dim=1)
            noiseZ = self.noiseEncoder(noiseInp)
            # print(noiseZ)
            # t=input()
            mu = noiseZ[:, 0:self.zSize] 
            logvar = noiseZ[:, self.zSize:] 
            var = logvar.mul(0.5).exp_()
            eps = torch.FloatTensor(var.size()).normal_()
            eps = eps.to(device)
            z = eps.mul(var).add_(mu)
            z = z.float().to(device)

            innerInp = torch.cat((obsFeat, z), dim=1)

            out = self.trajectoryDecoder(innerInp)
            pred = out.view(-1, predLength,2)

            pred[:,:,0] = (pred[:,:,0] + 1.0)*W*0.5
            pred[:,:,1] = (pred[:,:,1] + 1.0)*H*0.5
            
            return pred, {
                'mean': mu,
                'var': logvar
            }
        else:
            # create N trajectories
            z = torch.Tensor(params.dataset.num_traj, obsFeat.size(0), self.zSize)
            z.normal_(0, self.sigma)
            z = z.float().to(device)

            obsFeat = obsFeat.unsqueeze(0).repeat(params.dataset.num_traj,1,1)
            innerInp = torch.cat((obsFeat, z), dim=2)
    
            out = self.trajectoryDecoder(innerInp)
            pred = out.view(params.dataset.num_traj,-1, predLength,2)
            pred[:,:,:,0] = (pred[:,:,:,0] + 1.0)*W*0.5
            pred[:,:,:,1] = (pred[:,:,:,1] + 1.0)*H*0.5

            
            return pred