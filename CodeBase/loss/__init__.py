from .lossfunction import GoalTrajLoss, PairwiseDistanceLoss, TrajMSELoss
lossfunc = {
    'GoalTrajLoss': GoalTrajLoss,
    'pairwise_distance_loss': PairwiseDistanceLoss,
    'TrajMSELoss': TrajMSELoss
}

def createLossFunction(params):
    # the lossfunc must be a class with func __call__(self,pred,gt,otherinfo)
    return lossfunc[params.model.loss_function.name](**params.model.loss_function.kwargs)