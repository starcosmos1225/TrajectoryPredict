from torch.utils.data.sampler import Sampler
import numpy as np
import random
import copy


class CustomSampler(Sampler):
    def __init__(self, params, dataInfo):
        self.len = dataInfo
        # print(self.len)
        # print(type(self.len))
        # print("end shwo len")
    
    def __iter__(self):
        index = list(np.random.permutation(self.len))
        return iter(index)

    def __len__(self):
        return self.len


class intervalSampler(Sampler):
    def __init__(self, params, dataInfo, num_interval=99999999):
        self.intervalList = dataInfo
        self.len = dataInfo[-1]
        self.batchSize = params.batch_size
        self.num_interval = num_interval
        self.index = self.createIter()
        
    def createIter(self):
        intervalBlank = []
        tempInterval = 0
        for interval in self.intervalList:
            order = np.arange(tempInterval, interval)
            intervalBlank.append(list(np.random.permutation(order)))
            tempInterval = interval
        random.shuffle(intervalBlank)
        intervalBlank = intervalBlank[:self.num_interval]
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
        self.len = len(index)
        return index

    def __iter__(self):
        copyIndex = copy.deepcopy(self.index)
        self.index = self.createIter()
        return iter(copyIndex)

    def __len__(self):
        # print("get sampler length",flush=True)
        return self.len

samplerDict = {
    'custom': CustomSampler,
    'interval': intervalSampler,
}

def createSampler(params,samplerInfo=None, **kwargs):
    return samplerDict[params.sampler.name](params,samplerInfo,**kwargs)