from torch.nn import BCEWithLogitsLoss

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

