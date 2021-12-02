import torch
import torch.nn as nn
from .backbone.resnet import resnet18, resnet50, resnet101, resnet152, \
    resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2, \
        conv3x3

class LocalNet(nn.Module):
    def __init__(self, 
                image_classes,
                pred_len,
                ):
        super(LocalNet, self).__init__()
        self.baseModel = resnet18(in_channels=pred_len*image_classes, num_classes=pred_len*2)
        
    def forward(self, obs, otherInp, extraInp, params):
        _,_,_,localImage,initTraj = otherInp
        # localImage: b,num_traj, squence, num_class, H,W
        # ->b*num_traj, squence*num_class,H,W
        b,numTraj,squence, numClass,H,W = localImage.shape
        localImage = localImage.view(b*numTraj,squence*numClass,H,W)
        # ---resnet---->b*num_traj, pred_len*2
        delta =  self.baseModel(localImage)
        #->b,num_traj,pred_len,2
        delta = delta.view(b,numTraj,params.dataset.pred_len,2)
        pred = initTraj.detach() + delta
        pred = pred.permute(1,0,2,3)
        
        if self.training:
            return pred, None
        else:
            return pred