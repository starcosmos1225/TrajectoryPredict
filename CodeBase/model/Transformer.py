import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer.decoder import Decoder
from .transformer.multihead_attention import MultiHeadAttention
from .transformer.positional_encoding import PositionalEncoding
from .transformer.pointerwise_feedforward import PointerwiseFeedforward
from .transformer.encoder_decoder import EncoderDecoder
from .transformer.encoder import Encoder
from .transformer.encoder_layer import EncoderLayer
from .transformer.decoder_layer import DecoderLayer
from .transformer.batch import subsequent_mask
import numpy as np
import scipy.io
import os

import copy
import math

class IndividualTF(nn.Module):
    def __init__(self, enc_inp_size, dec_inp_size, dec_out_size, N=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1,mean=[0,0],std=[0,0]):
        super(IndividualTF, self).__init__()
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadAttention(h, d_model)
        ff = PointerwiseFeedforward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        self.mean=np.array(mean)
        self.std=np.array(std)
        self.model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                 c(ff), dropout), N),
            nn.Sequential(LinearEmbedding(enc_inp_size,d_model), c(position)),
            nn.Sequential(LinearEmbedding(dec_inp_size,d_model), c(position)),
            Generator(d_model, dec_out_size))
        print("init IndividualTF")
        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, obs, otherInp=None,extraInfo=None,params=None):
        if self.training:
            device = params.device
            obsVel,predVel = otherInp[0], otherInp[1]
            mean, std = extraInfo
            inp=(obsVel-mean)/std
            target=(predVel-mean)/std
            target_c=torch.zeros((target.shape[0],target.shape[1],1))
            target=torch.cat((target,target_c),-1)
            startOfSeq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(target.shape[0],1,1).to(device)

            decInp = torch.cat((startOfSeq, target), 1)
            srcAtt = torch.ones((inp.shape[0], 1,inp.shape[1]))
            trgAtt=subsequent_mask(decInp.shape[1]).repeat(decInp.shape[0],1,1)
            outVelocity = self.model.generator(self.model(inp,decInp,srcAtt,trgAtt))
            pred = outVelocity[:,:,:2]* std+mean
            pred = (outVelocity[:, 1:, 0:2] * std + mean).cumsum(dim=1) + obs[:,-1:,:]
            # position = obs[:,-1,:] # 4,2
            # for i in range(pred.shape[1]):
            #     pred[:,i,:] += position
            #     position = pred[:,i,:]
            return pred, [outVelocity]
        else:
            device = params.device
            obsVel,predVel = otherInp[0], otherInp[1]
            mean, std = extraInfo
            inp = (obsVel- mean) / std
            srcAtt = torch.ones((inp.shape[0], 1, inp.shape[1]))
            startOfSeq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(inp.shape[0], 1, 1).to(
                    device)
            decInp = startOfSeq
            for i in range(params.dataset.pred_len):
                trgAtt = subsequent_mask(decInp.shape[1]).repeat(decInp.shape[0], 1, 1)
                outVelocity = self.model.generator(self.model(inp,decInp,srcAtt,trgAtt))
                decInp = torch.cat((decInp, outVelocity[:, -1:, :]), 1)
            pred = (decInp[:, 1:, 0:2] * std + mean).cumsum(dim=1) + obs[:,-1:,:]
            return pred, None

class LinearEmbedding(nn.Module):
    def __init__(self, inp_size,d_model):
        super(LinearEmbedding, self).__init__()
        # lut => lookup table
        self.lut = nn.Linear(inp_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, out_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, out_size)

    def forward(self, x):
        return self.proj(x)