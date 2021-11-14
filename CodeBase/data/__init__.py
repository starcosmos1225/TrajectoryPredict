from .ImageTrajDataloader import createImageTrajDataloader
# from TrajDataloader import TrajDataloader

dataDict = {
    'imageTraj': createImageTrajDataloader,
    # 'trajectory': TrajDataloader,
}



def createDataLoader(params,type='train'):
    return dataDict[params.dataset.type](params,type)

