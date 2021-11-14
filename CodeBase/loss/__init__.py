from .lossfunction import GoalTrajLoss
lossfunc = {
    'GoalTrajLoss': GoalTrajLoss
}

def createLossFunction(params):
    # the lossfunc must be a class with func __call__(self,pred,gt,otherinfo)
    return lossfunc[params.model.loss_function.name](**params.model.loss_function.kwargs)