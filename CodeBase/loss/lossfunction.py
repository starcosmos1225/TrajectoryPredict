from torch.nn import BCEWithLogitsLoss

class GoalTrajLoss:

    def __init__(self,loss_scale):
        self.lossFunc = BCEWithLogitsLoss()
        self.lossScale = loss_scale
    
    def __call__(self,pred,gt,otherInfos):
        predGoalMap = otherInfos
        goalLoss = self.lossFunc(predGoalMap,gt) * self.lossScale
        trajLoss = self.lossFunc(pred,gt) * self.lossScale
        return goalLoss + trajLoss

