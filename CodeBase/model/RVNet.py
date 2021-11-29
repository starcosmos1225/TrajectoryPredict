import torch
from torch import nn
from .backbone.Linear import MLP


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
        self.squenceFeat = MLP(pred_len, squence_out_size, squence_feat)
        self.trajectoryFeat = MLP(squence_out_size*num_classes*base_out_size,
                                pred_len*2,
                                traj_feat)
    def forward(self, obs, otherInp, extraInp, params):
        _,_,_,relativeVector,initTraj = otherInp
        # semanticMap (b,num_traj,squence, num_class, 512,2)
        # initTraj (b,num_traj,squence,2)
        
        # ->b*num_traj, squence, num_class, 1024 -> 64

        b,numTraj, predLength, numClass, numSample,_ = relativeVector.shape
        relativeVector = relativeVector.view(b*numTraj,predLength,numClass,-1)
        # print(relativeVector.dtype)
        # t=input()
        relativeFeat = self.baseFeat(relativeVector)
        # ->b*num_traj, num_class*64 ,squence
        relativeFeat = relativeFeat.view(b*numTraj, predLength, -1)
        relativeFeat = relativeFeat.permute(0,2,1)
        # ->b*num_traj, num_class*64, squence_out
        relativeSquence = self.squenceFeat(relativeFeat)
        # ->b,num_traj, squence, 2
        relativeSquence = relativeSquence.view(b,numTraj,-1)
        delta = self.trajectoryFeat(relativeSquence).view(b,numTraj,predLength,2)
        pred = initTraj + delta
        pred = pred.permute(1,0,2,3)
        # print(pred.shape)
        # t=input()
        if self.training:
            return pred, None
        else:
            return pred

