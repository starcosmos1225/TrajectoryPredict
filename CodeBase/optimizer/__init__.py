from torch.optim import Adam
from .Noam import Noam
optimizer_dict = {
    'Adam': Adam,
    'Noam': Noam
}


def createOptimizer(params, model):
    return optimizer_dict[params.optim.name](model.parameters(), **params.optim.kwargs)