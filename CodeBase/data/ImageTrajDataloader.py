import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from .preprocessing import augmentData, createImagesDict
from .dataloader import SceneDataset
from .image_utils import createGaussianHeatmapTemplate, createDistMat, preprocessImageForSegmentation, pad, resize
from .sampler import createSampler
# from .__init__ import createSampler

def createImageTrajDataloader(params,type='train'):

    obsLength = params.dataset.obs_len
    predLength = params.dataset.pred_len
    if params.dataset.segmentation_model_fp is not None:
        semanticModel = torch.load(params.dataset.segmentation_model_fp,map_location='cpu')
        if params.dataset.use_features_only:
            semanticModel.segmentation_head = nn.Identity()
            # semanticClasses = 16  # instead of classes use number of feature_dim
    else:
        semanticModel = nn.Identity()
    # totalLength = obsLength + predLength
    if type=='train':
        trainDataPath = params.dataset.train_data_path
        trainImagePath = params.dataset.train_image_path
        valDataPath = params.dataset.val_data_path
        valImagePath = params.dataset.val_image_path
        if not trainDataPath.endswith('pkl') or not valDataPath.endswith('pkl'):
            raise ValueError('ImageTrajDataloader could only read the pkl data!')
        trainData = pd.read_pickle(trainDataPath)
        valData = pd.read_pickle(valDataPath)
        datasetName = params.dataset.dataset_name.lower()
        if datasetName == 'sdd':
            imageFileName = 'reference.jpg'
        elif datasetName == 'ind':
            imageFileName = 'reference.png'
        # elif datasetName == 'eth':
        #     imageFileName = 'oracle.png'
        else:
            raise ValueError('{} dataset is not supported'.format(datasetName))

        # ETH/UCY specific: Homography matrix is needed to convert pixel to world coordinates
        if datasetName == 'eth':
            homoMat = {}
            for scene in ['eth', 'hotel', 'students001', 'students003', 'uni_examples', 'zara1', 'zara2', 'zara3']:
                homoMat[scene] = torch.Tensor(np.loadtxt(f'data/eth_ucy/{scene}_H.txt'))
            segMask = True
        else:
            homoMat = None
            segMask = False

        # Load train images and augment train data and images
        trainData, trainImages = augmentData(trainData, image_path=trainImagePath, imageFile=imageFileName,
                                                                                  segMask=segMask)

        # Load val scene images
        valImages = createImagesDict(valData, imagePath=valImagePath, imageFile=imageFileName)

        # Preprocess images, in particular resize, pad and normalize as semantic segmentation backbone requires
        resize(trainImages, factor=params.dataset.resize, segMask=segMask)
        pad(trainImages, divisionFactor=params.dataset.division_factor)  # make sure that image shape is divisible by 32, for UNet segmentation
        preprocessImageForSegmentation(trainImages, segMask=segMask)

        resize(valImages, factor=params.dataset.resize, segMask=segMask)
        pad(valImages, divisionFactor=params.dataset.division_factor)  # make sure that image shape is divisible by 32, for UNet segmentation
        preprocessImageForSegmentation(valImages, segMask=segMask)

        # Create template
        size = int(4200 * params.dataset.resize)

        inputTemplate = createDistMat(size=size)
        inputTemplate = torch.Tensor(inputTemplate)

        gtTemplate = createGaussianHeatmapTemplate(size=size, kernlen=params.dataset.kernlen, nsig=params.dataset.nsig, normalize=False)
        gtTemplate = torch.Tensor(gtTemplate)

        # Initialize dataloaders
        trainDataset = SceneDataset(trainData, 
                                    resize=params.dataset.resize, 
                                    obsLength=obsLength,
                                    predLength=predLength,
                                    sceneImages=trainImages,
                                    inputTemplate=inputTemplate,
                                    gtTemplate=gtTemplate,
                                    waypoints = params.dataset.waypoints,
                                    semanticModel=semanticModel)
            # create sampler
        if 'getSamplerInfo' in dir(trainDataset):
            trainSamplerInfo = trainDataset.getSamplerInfo()
        else:
            trainSamplerInfo = len(trainDataset)
        trainSampler = createSampler(params,trainSamplerInfo)
        trainLoader = DataLoader(trainDataset, 
                                 batch_size=params.dataset.batch_size, 
                                sampler=trainSampler)

        valDataset = SceneDataset(valData, 
                                  resize=params.dataset.resize, 
                                  obsLength=obsLength,
                                  predLength=predLength,
                                  sceneImages=valImages,
                                  inputTemplate=inputTemplate,
                                  gtTemplate=gtTemplate,
                                  waypoints = params.dataset.waypoints,
                                  semanticModel=semanticModel)
        if 'getSamplerInfo' in dir(valDataset):
            valSamplerInfo = valDataset.getSamplerInfo()
        else:
            valSamplerInfo = len(valDataset)
        valSampler = createSampler(params,valSamplerInfo)
        # valSampler = createSampler(params,valDataset.getSamplerInfo())
        valLoader = DataLoader(valDataset, 
                               batch_size=params.dataset.batch_size,
                               sampler=valSampler)
        return trainLoader, valLoader
    elif type=='test':
        pass
    else:
        raise ValueError('ImageTrajDataloader  is not supported type:{}'.format(type))

