from torch.utils.data.sampler import Sampler
import numpy as np
class CustomSampler(Sampler):
    def __init__(self, params, dataInfo):
        self.len = dataInfo
        print(self.len)
        print(type(self.len))
        print("end shwo len")
    
    def __iter__(self):
        index = list(np.random.permutation(self.len))
        return iter(index)

    def __len__(self):
        return self.len


class intervalSampler(Sampler):
    def __init__(self, params, dataInfo):
        self.intervalList = dataInfo
        self.len = dataInfo[-1]
        self.batchSize = params.dataset.batch_size
    def __iter__(self):
        intervalBlank = []
        tempInterval = 0
        for interval in self.intervalList:
            order = np.arange(tempInterval, interval)
            intervalBlank.append(list(np.random.permutation(order)))
            tempInterval = interval
        index = []
        while len(intervalBlank)>0:
            id = np.random.randint(len(intervalBlank))
            for i in range(self.batchSize):
                index.append(intervalBlank[id].pop())
                if len(intervalBlank[id])==0:
                    for j in range(self.batchSize-i-1):
                        index.append(index[-1])
                    intervalBlank.pop(id)
                    break
            if len(index)>self.len*2:
                break
            print("{}/{}".format(len(index),self.len))
        return iter(index)

    def __len__(self):
        return self.len

samplerDict = {
    'custom': CustomSampler,
    'interval': intervalSampler,
}

def createSampler(params,samplerInfo=None):
    return samplerDict[params.dataset.sampler](params,samplerInfo)