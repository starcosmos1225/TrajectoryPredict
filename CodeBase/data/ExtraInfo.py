from data.image_utils import createDistMat
import torch
import numpy as np


def inputTemplate(params,dataLoaders):
    device = params.device
    size = int(4200* params.dataset.resize)
    inputTemplate = createDistMat(size=size)
    inputTemplate = torch.from_numpy(inputTemplate).float().to(device)
    return inputTemplate

def datasetMeanStd(params, dataLoaders):
    device = params.device
    train_dataset = dataLoaders[0].dataset
    mean=torch.cat((train_dataset[:]['obs'][:,1:,2:4],train_dataset[:]['pred'][:,:,2:4]),1).mean((0,1))
    std=torch.cat((train_dataset[:]['obs'][:,1:,2:4],train_dataset[:]['pred'][:,:,2:4]),1).std((0,1))
    means=[]
    stds=[]
    for i in np.unique(train_dataset[:]['dataset']):
        ind=train_dataset[:]['dataset']==i
        means.append(torch.cat((train_dataset[:]['obs'][ind, 1:, 2:4], train_dataset[:]['pred'][ind, :, 2:4]), 1).mean((0, 1)))
        stds.append(
            torch.cat((train_dataset[:]['obs'][ind, 1:, 2:4], train_dataset[:]['pred'][ind, :, 2:4]), 1).std((0, 1)))
    mean=torch.stack(means).mean(0).to(device)
    std=torch.stack(stds).mean(0).to(device)
    return mean,std

def constantFactor(params, dataLoaders):
    return {
        'resize': params.dataset.resize,
        'modelname': params.model.name
    }