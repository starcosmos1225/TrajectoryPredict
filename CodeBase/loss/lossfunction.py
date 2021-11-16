import torch
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
class GoalTrajLoss:

    def __init__(self,loss_scale):
        self.lossFunc = BCEWithLogitsLoss()
        self.lossScale = loss_scale
    
    def __call__(self,pred,gt,otherInp, otherOut):
        gtFutureMap = otherInp[1]
        predGoalMap = otherOut[1]
        predTrajMap = otherOut[0]
        goalLoss = self.lossFunc(predGoalMap,gtFutureMap) * self.lossScale
        trajLoss = self.lossFunc(predTrajMap,gtFutureMap) * self.lossScale
        return goalLoss + trajLoss


class PairwiseDistanceLoss:

    def __init__(self):
        self.lossFunc = F.pairwise_distance
    
    def __call__(self,pred,gt,otherInp, otherOut,extraInfo):
        mean, std = extraInfo
        gtPredVel = otherInp[2]
        predVel = otherOut[0]
        loss = self.lossFunc(predVel[:, :,0:2].contiguous().view(-1, 2),
                                       ((gtPredVel-mean)/std).contiguous().view(-1, 2)).mean() + \
                                        torch.mean(torch.abs(predVel[:,:,2]))

        return loss
