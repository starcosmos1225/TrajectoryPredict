from torch.optim import Adam

optimizer_dict = {
    'Adam': Adam
}


def createOptimizer(params, model):
    return optimizer_dict[params.optim.name](model.parameters(), **params.optim.kwargs)