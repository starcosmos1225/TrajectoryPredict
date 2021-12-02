import torch
from torch import nn
from .transformer.multihead_attention import MultiHeadAttention
from .transformer.pointerwise_feedforward import PointerwiseFeedforward
from .transformer.encoder_decoder import EncoderDecoder
from .transformer.encoder import Encoder
from .transformer.encoder_layer import EncoderLayer
from .backbone.Linear import MLP, ResMLP
from .backbone.NonOrderNet import NonOrderNet
from .backbone.resnet1d import resnet1d_18
import copy
import time


class RVNet(nn.Module):
    def __init__(self, 
                num_samples,
                num_classes,
                base_out_size,
                base_feat, 
                pred_len,
                squence_out_size,
                squence_feat,
                traj_feat,
                ):
        super(RVNet, self).__init__()
        self.baseFeat = MLP(num_samples*2, base_out_size, base_feat)
        self.squenceFeat = MLP(pred_len*num_classes*base_out_size, squence_out_size, squence_feat)
        self.trajectoryFeat = MLP(squence_out_size,
                                pred_len*2,
                                traj_feat)
    def forward(self, obs, otherInp, extraInp, params):
        _,_,_,relativeVector,initTraj = otherInp
        # semanticMap (b,num_traj,squence, num_class, 512,2)
        # initTraj (b,num_traj,squence,2)
        
        # ->b*num_traj, squence, num_class, 1024 -> 64

        b,numTraj, predLength, numClass, numSample,_ = relativeVector.shape
        relativeVector = relativeVector.view(b*numTraj,predLength,numClass,-1)
        relativeFeat = self.baseFeat(relativeVector)
        # ->b*num_traj, num_class*64*quence
        relativeFeat = relativeFeat.view(b*numTraj, -1)
        # relativeFeat = relativeFeat.permute(0,2,1)
        # ->b*num_traj, squence_out_size
        relativeSquence = self.squenceFeat(relativeFeat)
        # ->b,num_traj, squence, 2
        relativeSquence = relativeSquence.view(b,numTraj,-1)
        delta = self.trajectoryFeat(relativeSquence).view(b,numTraj,predLength,2)
        pred = initTraj.detach() + delta
        pred = pred.permute(1,0,2,3)
        # if self.training:
        #     return initTraj.permute(1,0,2,3), None
        # else:
        #     return initTraj.permute(1,0,2,3)
        if self.training:
            return pred, None
        else:
            return pred


class RVNetResdual(nn.Module):
    def __init__(self, 
                num_samples,
                num_classes,
                base_out_size,
                base_feat, 
                pred_len,
                squence_out_size,
                squence_feat,
                traj_feat,
                ):
        super(RVNetResdual, self).__init__()
        self.baseFeat = ResMLP(inp=num_classes*num_samples*2*pred_len, out=pred_len*2,scale=0.25,activate='ReLU',bn=True)
        
    def forward(self, obs, otherInp, extraInp, params):
        _,_,_,relativeVector,initTraj = otherInp
        
        b,numTraj, predLength, numClass, numSample,_ = relativeVector.shape
        relativeVector = relativeVector.view(b,numTraj,-1)
        delta = self.baseFeat(relativeVector).view(b,numTraj,predLength,2)
        pred = initTraj.detach() + delta
        pred = pred.permute(1,0,2,3)
        
        if self.training:
            return pred, None
        else:
            return pred

class RVNetConv1d(nn.Module):
    def __init__(self, 
                pred_len,
                kernel_size = 65,
                ):
        # num_classes,
        # in_channels=3,
        # kernel_size=64,
        # width_per_layer = [64,128,256,512],
        # zero_init_residual=False,
        # groups= 1,
        # width_per_group = 64,
        # replace_stride_with_dilation= None,
        # norm_layer= None,
        super(RVNetConv1d, self).__init__()
        self.baseFeat = resnet1d_18(in_channels=pred_len, num_classes=pred_len*2, kernel_size=kernel_size)
        
    def forward(self, obs, otherInp, extraInp, params):
        _,_,_,relativeVector,initTraj = otherInp
        
        b,numTraj, predLength, numClass, numSample,_ = relativeVector.shape
        relativeVector = relativeVector.view(b*numTraj,predLength,1,-1)

        delta = self.baseFeat(relativeVector).view(b,numTraj,predLength,2)
        pred = initTraj.detach() + delta
        pred = pred.permute(1,0,2,3)
        
        if self.training:
            return pred, None
        else:
            return pred



class RVNetTransformer(nn.Module):
    def __init__(self, 
                num_samples,
                class_out,
                class_feat,
                pred_len,
                pred_feat,
                num_classes,
                num_heads=8,
                d_ff=2048,
                dropout=0.1,
                encoder_layer=6,
                use_transformer=True,
                use_nonOrder=True
                ):
        super(RVNetTransformer, self).__init__()
        h = num_heads
        d_model = num_classes*class_out
        self.nonOrder = NonOrderNet(inp=2, out=class_out, feat=class_feat,type='max')
        c = copy.deepcopy
        attn = MultiHeadAttention(h, d_model)
        ff = PointerwiseFeedforward(d_model, d_ff, dropout)
        self.transformerEncoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), encoder_layer)
        self.decoder = MLP(d_model*pred_len,pred_len*2,pred_feat)
        self.useTransformer = use_transformer
        self.useNonOrder = use_nonOrder
        if not self.useNonOrder:
            self.nonOrder = MLP(2*num_samples, class_out, [num_samples, 2*class_out])
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, obs, otherInp, extraInp, params):
        _,_,_,relativeVector,initTraj = otherInp
        # if self.training:
        #     return initTraj.permute(1,0,2,3) , None
        # else:
        #     return initTraj.permute(1,0,2,3)
        b,numTraj, predLength, numClass, numSample,_ = relativeVector.shape
        if self.useNonOrder:
            relativeVector = relativeVector.view(-1, numSample,2)
            
        else:
            relativeVector = relativeVector.view(-1, numClass,numSample*2)
        # b*numTraj*predLength, numClass,out
        feat = self.nonOrder(relativeVector)  
        if self.useTransformer:
            feat = feat.view(b*numTraj, predLength, -1)
            feat = self.transformerEncoder(feat, None)
        feat = feat.view(b,numTraj, -1)
        feat = self.decoder(feat)
        delta = feat.view(b, numTraj, predLength,2)
        pred = initTraj.detach() + delta
        pred = pred.permute(1,0,2,3)

        if self.training:
            return pred , None
        else:
            return pred
        