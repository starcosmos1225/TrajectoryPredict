import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone.Linear import LinearEmbedding, MLP
import copy
import time


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
        if self.training:
            return pred, [outVelocity]
        else:
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
        if self.training:

            return pred, [outVelocity]
        else:
            return pred, None

class DoubleCVAELSTM(nn.Module):
    def __init__(self, inp_size, out_size, num_layers,z_size,sigma,
                   hidden_size=512):
        super(DoubleCVAELSTM, self).__init__()
        self.sigma = sigma

        self.encoder = LinearEmbedding(inp_size, hidden_size)
        self.decoder = LinearEmbedding(hidden_size + z_size, out_size)
        self.hiddenAmplifier = LinearEmbedding(hidden_size, hidden_size+z_size)
        self.CVAEEncoder = nn.Sequential(
                       MLP(hidden_size, z_size, hidden_size//2),
                       nn.AdaptiveAvgPool2d((1,z_size))
        )
        self.noiseEncoder = MLP(hidden_size + z_size , z_size * 2 , z_size)

        self.LSTM1 = nn.LSTM(hidden_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)

        self.LSTMCells = nn.ModuleList([ nn.LSTMCell(hidden_size + z_size,hidden_size + z_size) for _ in range(num_layers)])
        self.numLayers = num_layers

        self.hiddenSize = hidden_size
        self.zSize = z_size
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, obs, otherInp=None,extraInfo=None,params=None):
        
        device = params.device
        obsVel,predVel = otherInp[0], otherInp[1]
        mean, std = extraInfo
        inp=(obsVel-mean)/std
        vel = (predVel - mean)/std
        predLength = params.dataset.pred_len

        embedInp = self.encoder(inp)
        predFeature = self.encoder(vel)
        predFeature = self.CVAEEncoder(predFeature).squeeze(1)
        output,(obsH,obsC) = self.LSTM1(embedInp)
        hiddenStatus = []
        for i in range(self.numLayers):
            hiddenStatus.append((self.hiddenAmplifier(obsH[i]),self.hiddenAmplifier(obsC[i])))
        
        innerInp = output[:,-1,:]

        predlist = []
        if self.training:
            # create z noise
            noiseInp = torch.cat((innerInp, predFeature),dim=1)
            noiseZ = self.noiseEncoder(noiseInp)
            mu = noiseZ[:, 0:self.zSize] 
            logvar = noiseZ[:, self.zSize:] 
            var = logvar.mul(0.5).exp_()
            eps = torch.FloatTensor(var.size()).normal_()
            eps = eps.to(device)
            z = eps.mul(var).add_(mu)
            z = z.float().to(device)
            # end CVAE
            innerInp = torch.cat((innerInp, z), dim=1)
            predlist.append(innerInp)

            for i in range(predLength-1):
                for idx,cell in enumerate(self.LSTMCells):
                    hiddenStatus[idx] = cell(innerInp,hiddenStatus[idx])
                    innerInp = hiddenStatus[idx][0]
                predlist.append(innerInp)
            predlist = torch.stack(predlist,dim=1)
            outVelocity = self.decoder(predlist)
            pred = outVelocity[:,:,:2]* std+mean
            pred = pred.cumsum(dim=1) + obs[:,-1:,:]
            if self.training:
                return pred, {
                    'mean': mu,
                    'logVar': logvar,
                    'predVel': outVelocity,
                }
        else:
            # create N trajectories
            predSamples = []
            for _ in range(params.dataset.num_traj):
                start = time.time()
                predlist = []
                z = torch.Tensor(innerInp.size(0), self.hiddenSize)
                z.normal_(0, self.sigma)
                z = z.float().to(device)
                hiddenStatusCopy = []
                for h,c in hiddenStatus:
                    hiddenStatusCopy.append((h.clone(),c.clone()))
                innerInpCopy = torch.cat((innerInp.clone(), z), dim=1)
                predlist.append(innerInpCopy)
                # print("lstm1 time:{}".format(time.time()-start))
                start= time.time()
                for i in range(predLength-1):
                    for idx,cell in enumerate(self.LSTMCells):
                        hiddenStatusCopy[idx] = cell(innerInpCopy,hiddenStatusCopy[idx])
                        innerInpCopy = hiddenStatusCopy[idx][0]
                    predlist.append(innerInpCopy)
                predlist = torch.stack(predlist,dim=1)
                outVelocity = self.decoder(predlist)
                # print("cal time:{}".format(time.time()-start))
                start= time.time()
                pred = outVelocity[:,:,:2]* std+mean
                pred = pred.cumsum(dim=1) + obs[:,-1:,:]
                predSamples.append(pred)
                # print("post time:{}".format(time.time()-start))
                
            return torch.stack(predSamples,dim=0), None


class DoubleLSTMPure(nn.Module):
    def __init__(self, inp_size, out_size, num_layers,
                   hidden_size=512):
        super(DoubleLSTMPure, self).__init__()

        self.encoder = LinearEmbedding(inp_size, hidden_size)
        self.decoder = LinearEmbedding(hidden_size, out_size)

        self.LSTM1 = nn.LSTM(hidden_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        
        self.LSTM2 = nn.LSTM(hidden_size, 
                             hidden_size, 
                             num_layers, 
                             batch_first=True)

        # self.LSTMCells = nn.ModuleList([ nn.LSTMCell(hidden_size,hidden_size) for _ in range(num_layers)])
        # self.numLayers = num_layers

        self.hiddenSize = hidden_size
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, obs, otherInp=None,extraInfo=None,params=None):
        obsVel,predVel = otherInp[0], otherInp[1]
        mean, std = extraInfo
        inp=(obsVel-mean)/std

        predLength = params.dataset.pred_len

        embedInp = self.encoder(inp)
        output,(obsH,obsC) = self.LSTM1(embedInp)
        # hiddenStatus = []
        # for i in range(self.numLayers):
        #     hiddenStatus.append((obsH[i],obsC[i]))
        predFeat = output[:,-1:,:]
        innerInp = output[:,-1:,:].repeat(1,predLength-1,1)
        outputFeat,_ = self.LSTM2(innerInp,(obsH,obsC))
        # print(predFeat.shape)
        # print(outputFeat.shape)
        predFeat = torch.cat((predFeat,outputFeat),dim=1)
        # print(predFeat.shape)
        outVelocity = self.decoder(predFeat)

        pred = outVelocity[:,:,:2]* std+mean
        pred = pred.cumsum(dim=1) + obs[:,-1:,:]
        if self.training:
            return pred, {
                'predVel': outVelocity
            }
        else:
            return pred, None

class DoubleLSTMGoal(nn.Module):
    def __init__(self, inp_size, out_size, num_layers,
                   hidden_size=512):
        super(DoubleLSTMGoal, self).__init__()

        self.encoder = LinearEmbedding(inp_size, hidden_size)
        self.decoder = LinearEmbedding(hidden_size, out_size)
        self.goalDecoder = LinearEmbedding(hidden_size, 2)
        self.goalEncoder = LinearEmbedding(2, hidden_size)
        self.LSTM1 = nn.LSTM(hidden_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        
        self.LSTM2 = nn.LSTM(hidden_size, 
                             hidden_size, 
                             num_layers, 
                             batch_first=True)
        
        self.LSTMGoal = nn.LSTM(hidden_size,
                                hidden_size,
                                num_layers,
                                batch_first=True)

        # self.LSTMCells = nn.ModuleList([ nn.LSTMCell(hidden_size,hidden_size) for _ in range(num_layers)])
        # self.numLayers = num_layers

        self.hiddenSize = hidden_size
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, obs, otherInp=None,extraInfo=None,params=None):
        obsVel,predVel = otherInp[0], otherInp[1]
        
        
        mean, std = extraInfo
        inp=(obsVel-mean)/std
            

        predLength = params.dataset.pred_len

        embedInp = self.encoder(inp)
        output,(obsH,obsC) = self.LSTM1(embedInp)

        predFeat = output[:,-1:,:]

        goalFeat, _ = self.LSTMGoal(embedInp)
        goal = self.goalDecoder(goalFeat[:,-1:,:])
        # print(gtGoalVel.shape)
        # t=input()
        if self.training:
            gtPredVel = otherInp[2]
            gtGoalVel = gtPredVel.cumsum(dim=1)[:,-1:,:]
            gtGoalVel = (gtGoalVel - mean) / std
            innerInp = self.goalEncoder(gtGoalVel).repeat(1,predLength-1,1)
        else:
            
            innerInp = self.goalEncoder(goal).repeat(1,predLength-1,1)
        # innerInp = output[:,-1:,:].repeat(1,predLength-1,1)
        # print(innerInp.shape)
        outputFeat,_ = self.LSTM2(innerInp,(obsH,obsC))

        predFeat = torch.cat((predFeat,outputFeat),dim=1)

        outVelocity = self.decoder(predFeat)

        pred = outVelocity[:,:,:2]* std+mean
        pred = pred.cumsum(dim=1) + obs[:,-1:,:]
        if self.training:
            return pred, {
                    'predVel': outVelocity,
                    'goalVel': goal
                }
        else:
            return pred, None
