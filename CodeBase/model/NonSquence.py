import torch
from torch import nn
from .backbone.Linear import LinearEmbedding, MLP


class CVAEDoubleMLP(nn.Module):
    def __init__(self, pred_len,inp_size, out_size, num_layers,z_size,sigma,
                   hidden_size=512):
        super(CVAEDoubleMLP, self).__init__()
        self.sigma = sigma

        self.obsEncoder = nn.Sequential(
                            LinearEmbedding(inp_size, hidden_size),
                            MLP(hidden_size, hidden_size, hidden_size//2),
                            nn.AdaptiveAvgPool2d((1,hidden_size))
                             )
        self.predEncoder = nn.Sequential(
                       LinearEmbedding(inp_size, hidden_size),
                       MLP(hidden_size, hidden_size, hidden_size//2),
                       nn.AdaptiveAvgPool2d((1,hidden_size))
        )
        self.decoder = LinearEmbedding(hidden_size + z_size, out_size)
        # self.hiddenAmplifier = LinearEmbedding(hidden_size, hidden_size+z_size)
        
        # self.CVAEDecoder = LinearEmbedding(hidden_size + z_size, hidden_size)
        self.noiseEncoder = MLP(hidden_size*2 , z_size * 2 , z_size)

        # self.LSTM2 = nn.LSTM(hidden_size + z_size,
        #                     hidden_size + z_size,
        #                     num_layers,
        #                     batch_first=True)
        self.trajectoryDecoder = nn.Sequential(
                        MLP(hidden_size + z_size, hidden_size+z_size, (hidden_size+z_size)//2),
                        LinearEmbedding(hidden_size+z_size,out_size))
                        # LinearEmbedding(hidden_size+z_size, 2*pred_len))
    

        self.numLayers = num_layers

        self.hiddenSize = hidden_size
        self.zSize = z_size
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, obs, otherInp=None,extraInfo=None,params=None):
        
        device = params.device
        obsVel = otherInp[0]
        mean, std = extraInfo
        inp=(obsVel-mean)/std
        # inp = obs - obs[:,:1,:]
        # vel = otherInp[0]
        # vel = vel - obs[:,:1,:]
        predLength = params.dataset.pred_len
        # inp (batch, s, 2) -> (batch,s, hiddensize)
        obsFeat = self.obsEncoder(inp)[:,-1,:]
        

        if self.training:
            # create z noise
            # (batch 2* hiddensize)
            gtPredVel =  otherInp[2]
            vel = (gtPredVel - mean)/std
            predFeat = self.predEncoder(vel)[:,-1,:]
            noiseInp = torch.cat((obsFeat, predFeat),dim=1)
            noiseZ = self.noiseEncoder(noiseInp)
            mu = noiseZ[:, 0:self.zSize] 
            logvar = noiseZ[:, self.zSize:] 
            var = logvar.mul(0.5).exp_()
            eps = torch.FloatTensor(var.size()).normal_()
            eps = eps.to(device)
            z = eps.mul(var).add_(mu)
            z = z.float().to(device)
            # end CVAE
            innerInp = torch.cat((obsFeat, z), dim=1).unsqueeze(1).repeat(1,predLength,1)
            # out = self.trajectoryDecoder(innerInp)
            # innerInp = torch.cat((obsFeat, z), dim=1)
            # b, h+z->b,2*predLength
            out = self.trajectoryDecoder(innerInp)
            outVelocity = out.view(-1, predLength,2)
            # outVelocity = self.decoder(out)
            pred = outVelocity[:,:,:2]*std+mean
            pred = pred.cumsum(dim=1) + obs[:,-1:,:]
            
            return pred, {
                'mean': mu,
                'var': logvar,
                'futureTraj': outVelocity,
            }
        else:
            # create N trajectories
            z = torch.Tensor(params.dataset.num_traj, obsFeat.size(0), self.zSize)
            z.normal_(0, self.sigma)
            z = z.float().to(device)

            obsFeat = obsFeat.unsqueeze(0).repeat(params.dataset.num_traj,1,1)
            innerInp = torch.cat((obsFeat, z), dim=2)
            # num_traj, batch, hiddensize*2->num_traj, batch,pred, hiddensize
            
            innerInp = innerInp.unsqueeze(2).repeat(1,1, predLength,1)
            innerInp = innerInp.view(-1,predLength,self.hiddenSize + self.zSize)
            
            
            out = self.trajectoryDecoder(innerInp)
            # print(out.shape)
            # t=input()
            outVelocity = out.view(params.dataset.num_traj,-1, predLength,2)
            # outVelocity = self.decoder(out)
            # print(outVelocity.shape)
            # t=input()
            pred = outVelocity * std+mean
            pred = pred.cumsum(dim=2) + obs.unsqueeze(0)[:,:,-1:,:]
            # pred = outVelocity[:,:,:,:2] + obs.unsqueeze(0)[:,:,:1,:]
            
            return pred
