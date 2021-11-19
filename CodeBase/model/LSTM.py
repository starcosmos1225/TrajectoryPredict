import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone.Linear import LinearEmbedding

class SingleLSTM(nn.Module):
    def __init__(self, inp_size, out_size, num_layers,
                   hidden_size=512):
        super(SingleLSTM, self).__init__()

        self.encoder = LinearEmbedding(inp_size, hidden_size)
        self.decoder = LinearEmbedding(hidden_size, out_size)
        self.LSTMCells = nn.ModuleList([ nn.LSTMCell(hidden_size,hidden_size) for _ in range(num_layers)])
        self.numLayers = num_layers

        self.hiddenSize = hidden_size
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def initHiddenStatus(self,batchSize, hiddenSize, device):
        hiddenList = []
        for _ in range(self.numLayers):
            h = torch.zeros((batchSize,hiddenSize)).to(device)
            c = torch.zeros((batchSize,hiddenSize)).to(device)
            hiddenList.append((h,c))
        return hiddenList

    def forward(self, obs, otherInp=None,extraInfo=None,params=None):
        if self.training:
            device = params.device
            batchSize = obs.shape[0]
            obsVel,predVel = otherInp[0], otherInp[1]
            mean, std = extraInfo
            inp=(obsVel-mean)/std
            # target=(predVel-mean)/std

            obsLength = params.dataset.obs_len
            predLength = params.dataset.pred_len

            embedInp = self.encoder(inp)
            hiddenStatus = self.initHiddenStatus(batchSize,self.hiddenSize,device)
            predlist = []
            innerInp = None
            for i in range(obsLength-1):
                innerInp = embedInp[:,i,:]
                for idx,cell in enumerate(self.LSTMCells):
                    hiddenStatus[idx] = cell(innerInp,hiddenStatus[idx])
                    innerInp = hiddenStatus[idx][0]# pred = self.decoder(hiddenStatus[0]).unsqueeze(1) # batch,1,pred
                predlist.append(innerInp)
            for i in range(predLength):
                for idx,cell in enumerate(self.LSTMCells):
                    hiddenStatus[idx] = cell(innerInp,hiddenStatus[idx])
                    innerInp = hiddenStatus[idx][0]
                    # print("innerInp.shape:{}".format(innerInp.shape))
                predlist.append(innerInp)
            predlist = torch.stack(predlist,dim=1)
            predlist = self.decoder(predlist)
            outVelocity = predlist[:, obsLength-1:,:]
            # print("velocity shape:{}".format(outVelocity.shape))

            pred = outVelocity[:,:,:2]* std+mean
            # print(pred.shape,obs[:,-1:,:].shape)
            pred = pred.cumsum(dim=1) + obs[:,-1:,:]

            return pred, [outVelocity]
        else:
            device = params.device
            batchSize = obs.shape[0]
            obsVel,predVel = otherInp[0], otherInp[1]
            mean, std = extraInfo
            inp=(obsVel-mean)/std

            obsLength = params.dataset.obs_len
            predLength = params.dataset.pred_len

            embedInp = self.encoder(inp)
            hiddenStatus = self.initHiddenStatus(batchSize,self.hiddenSize,device)
            predlist = []
            innerInp = None
            for i in range(obsLength-1):
                innerInp = embedInp[:,i,:]
                for idx,cell in enumerate(self.LSTMCells):
                    hiddenStatus[idx] = cell(innerInp,hiddenStatus[idx])
                    innerInp = hiddenStatus[idx][0]
                predlist.append(innerInp)
            for i in range(predLength):
                for idx,cell in enumerate(self.LSTMCells):
                    hiddenStatus[idx] = cell(innerInp,hiddenStatus[idx])
                    innerInp = hiddenStatus[idx][0]
                predlist.append(innerInp)
            predlist = torch.stack(predlist,dim=1)
            predlist = self.decoder(predlist)
            outVelocity = predlist[:, obsLength-1:,:]

            pred = outVelocity[:,:,:2]* std+mean
            pred = pred.cumsum(dim=1) + obs[:,-1:,:]
            
            return pred, None

class DoubleLSTM(nn.Module):
    def __init__(self, inp_size, out_size, num_layers,
                   hidden_size=512):
        super(DoubleLSTM, self).__init__()

        self.encoder = LinearEmbedding(inp_size, hidden_size)
        self.decoder = LinearEmbedding(hidden_size, out_size)

        self.LSTM1 = nn.LSTM(hidden_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)

        self.LSTMCells = nn.ModuleList([ nn.LSTMCell(hidden_size,hidden_size) for _ in range(num_layers)])
        self.numLayers = num_layers

        self.hiddenSize = hidden_size
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, obs, otherInp=None,extraInfo=None,params=None):
        if self.training:
            obsVel,predVel = otherInp[0], otherInp[1]
            mean, std = extraInfo
            inp=(obsVel-mean)/std

            predLength = params.dataset.pred_len

            embedInp = self.encoder(inp)
            predlist = []
            output,(obsH,obsC) = self.LSTM1(embedInp)
            hiddenStatus = []
            for i in range(self.numLayers):
                hiddenStatus.append((obsH[i],obsC[i]))
            predlist.append(output[:,-1,:])
            innerInp = output[:,-1,:]
            for i in range(predLength-1):
                for idx,cell in enumerate(self.LSTMCells):
                    hiddenStatus[idx] = cell(innerInp,hiddenStatus[idx])
                    innerInp = hiddenStatus[idx][0]
                predlist.append(innerInp)
            predlist = torch.stack(predlist,dim=1)
            outVelocity = self.decoder(predlist)

            pred = outVelocity[:,:,:2]* std+mean
            pred = pred.cumsum(dim=1) + obs[:,-1:,:]

            return pred, [outVelocity]
        else:
            obsVel,predVel = otherInp[0], otherInp[1]
            mean, std = extraInfo
            inp=(obsVel-mean)/std

            predLength = params.dataset.pred_len

            embedInp = self.encoder(inp)
            predlist = []
            output,(obsH,obsC) = self.LSTM1(embedInp)
            hiddenStatus = []
            for i in range(self.numLayers):
                hiddenStatus.append((obsH[i],obsC[i]))
            predlist.append(output[:,-1,:])
            innerInp = output[:,-1,:]
            for i in range(predLength-1):
                for idx,cell in enumerate(self.LSTMCells):
                    hiddenStatus[idx] = cell(innerInp,hiddenStatus[idx])
                    innerInp = hiddenStatus[idx][0]
                predlist.append(innerInp)
            predlist = torch.stack(predlist,dim=1)
            outVelocity = self.decoder(predlist)

            pred = outVelocity[:,:,:2]* std+mean
            pred = pred.cumsum(dim=1) + obs[:,-1:,:]

            return pred, None

