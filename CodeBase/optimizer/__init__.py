from torch.optim import Adam
from .Noam import Noam
from .sgd_agc import SGD_AGC

optimizer_dict = {
    'Adam': Adam,
    'Noam': Noam,
    'SGD_AGC': SGD_AGC,
}


def createOptimizer(params, model):
    return optimizer_dict[params.optim.name](model.parameters(), **params.optim.kwargs)