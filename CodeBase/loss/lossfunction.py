import torch
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
class GoalTrajLoss:

    def __init__(self,loss_scale):
        self.lossFunc = BCEWithLogitsLoss()
        self.lossScale = loss_scale
        self.lossFuncGoal = F.mse_loss
    
    def __call__(self,pred,gt,otherInp, otherOut, extraInfo):
        gtFutureMap = otherInp[1]
        # gtWaypointMap = otherInp[2]
        predGoalMap = otherOut[1]
        predTrajMap = otherOut[0]
        if predGoalMap is not None:
            goalLoss = self.lossFunc(predGoalMap,gtFutureMap) * self.lossScale
            # goalLoss = self.lossFuncGoal(predGoalMap,gtWaypointMap) * self.lossScale
        else:
            goalLoss = 0.0
        trajLoss = self.lossFunc(predTrajMap,gtFutureMap) * self.lossScale
        # print("goalLoss:{} trajLoss:{}".format(goalLoss, trajLoss))
        return goalLoss + trajLoss


class PairwiseDistanceLoss:

    def __init__(self, use_goal=False, weight_goal=1.0, use_kdl=False, weight_kdl=1.0):
        self.lossFunc = F.pairwise_distance
        self.useGoal = use_goal
        self.useKdl = use_kdl
        self.weightGoal = weight_goal
        self.weightKdl = weight_kdl
        if use_goal:
            self.lossGoal = F.mse_loss
    def __call__(self,pred,gt,otherInp, otherOut,extraInfo):
        
        mean, std = extraInfo
        gtPredVel = otherInp[2]
        
        predVel = otherOut['predVel']
        
        loss = self.lossFunc(predVel[:, :,0:2].contiguous().view(-1, 2),
                                       ((gtPredVel-mean)/std).contiguous().view(-1, 2)).mean() + \
                                        torch.mean(torch.abs(predVel[:,:,2]))
        if self.useGoal:
            # gtPredVel = otherInp[2]
            gtGoalVel = gtPredVel.cumsum(dim=1)[:,-1:,:]
            gtGoalVel = (gtGoalVel - mean) / std
            goalVel = otherOut['goalVel']
            # print(gtGoalVel.shape)
            # print(goalVel.shape)
            # t=input()
            goalLoss = self.lossGoal(gtGoalVel,goalVel) * self.weightGoal
        else:
            goalLoss = 0.0
        if self.useKdl:
            logVar = otherOut['logVar']
            mean = otherOut['mean']
            kldLoss = -0.5 * torch.sum(1 + logVar - mean.pow(2) - logVar.exp())
        else:
            kldLoss = 0.0
        return loss + goalLoss + kldLoss

class TrajMSELoss:
    def __init__(self):
        self.lossFunc = F.mse_loss
    
    def __call__(self,pred,gt,otherInp, otherOut,extraInfo):
        assert gt.shape[0] ==1
        if pred.shape[0] != gt.shape[0]:
            gt = gt.repeat(pred.shape[0],1,1,1)
        return self.lossFunc(pred, gt)

class VarietyLoss:
    def __init__(self):
        self.lossFunc = F.mse_loss
    
    def __call__(self,pred,gt,otherInp,otherOut,extraInfo):
        assert gt.shape[0] ==1
        predLength = pred.shape[2]
        dist2 = torch.sum((pred.detach()-gt.detach())**2, dim=-1,keepdim=True)
        dist2 = torch.sum(dist2, dim=-2,keepdim=True)
        index = torch.argmin(dist2,dim=0,keepdim=True)
        index = index.repeat(1,1,predLength,2)
        bestTraj = pred.gather(0,index)
        
        return self.lossFunc(bestTraj,gt)
        

class TrajCVAELoss:
    def __init__(self, traj_weight=1.0, kld_weight=1.0, rcl_weight=1.0):
        self.trajWeight = traj_weight
        self.kldWeight = kld_weight
        self.reconstructWeight = rcl_weight
        self.lossFunc = F.mse_loss

    def __call__(self,pred,gt, otherInp, otherOut, extraInfo):
        
        if 'modelname' in extraInfo and extraInfo['modelname'] == 'PECNet':
            gt, obs = otherInp[0], otherInp[1]
            gt = gt-obs[:,:1,:]
        else:
            # trajLoss = self.lossFunc(pred, gt)
            obsVel,gtVel = otherInp[0], otherInp[2]
            mean, std = extraInfo
            gt = (gtVel - mean) / std
                 
        if 'goal' in otherOut:
            goal = otherOut['goal']
            gtGoal = gt[:,-1,:]
            gtFutureTraj = gt[:,:-1,:].view(gt.shape[0],-1)
            reconstructLoss = self.lossFunc(goal,gtGoal)
        else:
            gtFutureTraj = gt.view(gt.shape[0],-1)
            reconstructLoss = 0.0
        mean, logVar,futureTraj = otherOut['mean'], \
                                  otherOut['var'], \
                                  otherOut['futureTraj']
        

        futureTraj = futureTraj.view(futureTraj.shape[0],-1)
        trajLoss = self.lossFunc(gtFutureTraj, futureTraj)
        kldLoss = -0.5 * torch.sum(1 + logVar - mean.pow(2) - logVar.exp())

        return trajLoss*self.trajWeight + kldLoss* self.kldWeight \
        + reconstructLoss* self.reconstructWeight
    
    # def calculate_loss(self,)

        
