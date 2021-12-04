# from TrajDataloader import TrajDataloader
from torch.utils.data import DataLoader
from .sampler import CustomSampler, intervalSampler
from .datasets.SceneDataset import SceneDataset
from .datasets.TrajDataset import TrajDataset
from .ExtraInfo import inputTemplate, datasetMeanStd, constantFactor
from .collate_fn import scene_collate,traj_collate, \
    PECtraj_collate, future_collate
datasetDict = {
    'scene_dataset': SceneDataset,
    'traj_dataset': TrajDataset,
}
samplerDict = {
    'custom': CustomSampler,
    'interval': intervalSampler,
}

collateDict = {
    'scene_collate': scene_collate,
    'traj_collate': traj_collate,
    'PECtraj_collate': PECtraj_collate,
    'future_collate': future_collate
}

extraDict = {
    'inputTemplate': inputTemplate,
    'datasetMeanStd': datasetMeanStd,
    'constantFactor': constantFactor
}

def createSampler(params,samplerInfo=None, **kwargs):
    return samplerDict[params.sampler.name](params,samplerInfo,**kwargs)



def createDataLoader(params,type='train'):
    datasetClass = datasetDict[params.dataset_name]
    collate_fn = collateDict[params.collate_name]
    if type=='train':
        # Initialize dataloaders
        
        trainDataset = datasetClass(params, mode='train')
            # create sampler
        if 'getSamplerInfo' in dir(trainDataset):
            trainSamplerInfo = trainDataset.getSamplerInfo()
        else:
            trainSamplerInfo = len(trainDataset)
        trainSampler = createSampler(params,trainSamplerInfo,**params.sampler.trainkwargs)
        trainLoader = DataLoader(trainDataset, 
                                 batch_size=params.batch_size, 
                                sampler=trainSampler,
                                num_workers=params.num_workers,
                                # pin_memory=True,
                                prefetch_factor=params.prefetch_factor,
                                collate_fn=collate_fn)

        valDataset = datasetClass(params,mode='val')
        if 'getSamplerInfo' in dir(valDataset):
            valSamplerInfo = valDataset.getSamplerInfo()
        else:
            valSamplerInfo = len(valDataset)
        valSampler = createSampler(params,valSamplerInfo,**params.sampler.valkwargs)
        # valSampler = createSampler(params,valDataset.getSamplerInfo())
        valLoader = DataLoader(valDataset, 
                               batch_size=params.batch_size,
                               sampler=valSampler,
                               num_workers=params.num_workers,
                               prefetch_factor=params.prefetch_factor,
                            #    pin_memory=True,
                               collate_fn=collate_fn)
        return trainLoader, valLoader
    elif type=='test':
        valDataset = datasetClass(params,mode='test')
        if 'getSamplerInfo' in dir(valDataset):
            valSamplerInfo = valDataset.getSamplerInfo()
        else:
            valSamplerInfo = len(valDataset)
        valSampler = createSampler(params,valSamplerInfo,**params.sampler.valkwargs)
        # valSampler = createSampler(params,valDataset.getSamplerInfo())
        valLoader = DataLoader(valDataset, 
                               batch_size=params.batch_size,
                               sampler=valSampler,
                               num_workers=params.num_workers,
                               prefetch_factor=params.prefetch_factor,
                            #    pin_memory=True,
                               collate_fn=collate_fn)
        return valLoader
    else:
        raise ValueError('ImageTrajDataloader  is not supported type:{}'.format(type))

def createExtraInfo(params, dataloaders):
    if 'extra_info_name' not in params or params.extra_info_name not in extraDict:
        return None
    return extraDict[params.extra_info_name](params, dataloaders)