import torch
import torch.nn as nn
from torch.nn import Parameter
#import torch.jit as jit
import warnings
from collections import namedtuple
from typing import List, Tuple
from torch import Tensor
import numbers
import math
DEVICE = 'cpu'
'''
Perfome a SCF+GRU structure
'''
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ir = Parameter(torch.randn( input_size))
        self.weight_hr = Parameter(torch.randn(hidden_size))
        self.bias_ir = Parameter(torch.randn(1))
        self.bias_hr = Parameter(torch.randn(1))
        self.weight_iz = Parameter(torch.randn(input_size))
        self.weight_hz = Parameter(torch.randn(hidden_size))
        self.bias_iz = Parameter(torch.randn(1))
        self.bias_hz = Parameter(torch.randn(1))
        self.weight_in = Parameter(torch.randn(input_size))
        self.weight_hn = Parameter(torch.randn(hidden_size))
        self.bias_in = Parameter(torch.randn(1))
        self.bias_hn = Parameter(torch.randn(1))

    #@jit.script_method
    def forward(self, input_x, hidden):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        rt = torch.sigmoid(torch.mm(input_x,self.weight_ir)+self.bias_ir+torch.mm(hidden,self.weight_hr)+self.bias_hr)
        zt = torch.sigmoid(torch.mm(input_x,self.weight_iz)+self.bias_iz+torch.mm(hidden,self.weight_hz)+self.bias_hz)
        nt = torch.tanh(torch.mm(input_x,self.weight_in)+self.bias_in+rt*(torch.mm(hidden,self.weight_hn)+self.bias_hn))
        ht = (1-zt)*nt + zt*hidden
        return ht

class SCF_GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size,radius_range,social_pooling_size):
        super(SCF_GRUCell, self).__init__()
        self.radius_range = radius_range
        self.social_pooling_size = social_pooling_size
        self.radius_step = (self.radius_range[1]-self.radius_range[0])/social_pooling_size[0]
        self.theta_step = 2*math.pi/social_pooling_size[1]
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ir = Parameter(torch.randn( input_size))
        self.weight_hr = Parameter(torch.randn(hidden_size))
        self.bias_ir = Parameter(torch.randn(1))
        self.bias_hr = Parameter(torch.randn(1))
        self.weight_iz = Parameter(torch.randn(input_size))
        self.weight_hz = Parameter(torch.randn(hidden_size))
        self.bias_iz = Parameter(torch.randn(1))
        self.bias_hz = Parameter(torch.randn(1))
        self.weight_in = Parameter(torch.randn(input_size))
        self.weight_hn = Parameter(torch.randn(hidden_size))
        self.bias_in = Parameter(torch.randn(1))
        self.bias_hn = Parameter(torch.randn(1))
        self.fc = nn.Sequential(nn.Linear(self.social_pooling_size[0]*self.social_pooling_size[1]*self.hidden_size,self.hidden_size),
                                nn.ReLU())
                            
    def compute_dist(self,loc_a,loc_b):
      '''
      loc_a:tensor(2)
      loc_b:tensor(2)
      return: distance (a-b) tensor(1) float
      '''
      return torch.norm(loc_a-loc_b)
    
    def compute_theta(self,loc_a,loc_b):
      '''
      loc_a:tensor(2)
      loc_b:tensor(2)
      return: angle (b-a) tensor(1) float(0~2pi)
      '''
      c = loc_b-loc_a
      dist = torch.norm(c)
      costheta = c[0]/dist
      if (c[1]<0):
        theta = 2*math.pi - costheta.acos()
      else:
        theta = costheta.acos()
      return theta
    #@jit.script_method
    def forward(self, loc_agent,loc_others,loc_other_index,feature_img,f_vel, hiddens,hidden):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        '''
        loca_agent:tensor(2)
        loc_others:tensor(n-1,2)
        loc_other_index:list of other agents' index
        feature_img: tensor(32,80,80)
        f_vel:tensor(16)
        hiddens:tensor(n,48)
        hidden:tensor(48)
        '''
        H = feature_img.shape[1]
        W = feature_img.shape[2]
        nums_feature = feature_img.shape[1]
        u = int(H/2-int(loc_agent[1]))
        v = int(loc_agent[0])
        # feature_agent:(32)
        feature_agent = feature_img[ :, u, v]
        # sp: tensor(6,6,48)
        sp = torch.zeros((self.social_pooling_size[0],self.social_pooling_size[1],self.hidden_size), device=torch.device(DEVICE))
        # sp_c: count the numbers in (6,6)
        sp_c = torch.zeros((self.social_pooling_size[0],self.social_pooling_size[1]), device=torch.device(DEVICE))
        # print(loc_others.shape[0])
        # print("len:{}".format(len(loc_other_index)))
        # t=input
        for i in range(loc_others.shape[0]):
          # loc:tensor(2)
          loc = loc_others[i]
          # dist:tensor(1)
          dist = self.compute_dist(loc, loc_agent)
          if self.radius_range[0] <= dist <= self.radius_range[1]:
            theta = self.compute_theta(loc_agent, loc)
            u = int((dist-self.radius_range[0])//self.radius_step)
            v = int((theta//self.theta_step))
            sp[u,v] += hiddens[loc_other_index[i]]
            sp_c[u,v] += 1
        for i in range(self.social_pooling_size[0]):
          for j in range(self.social_pooling_size[1]):
            if sp_c[i][j] > 1.0:
              sp[i][j] = sp[i][j]/sp_c[i][j]
        #(6,6,48)->(6*6*48)
        sp = sp.view(self.social_pooling_size[0]*self.social_pooling_size[1]*self.hidden_size)
        #(6*6*48)->(48)
        fsp = self.fc(sp)
        input_x = torch.cat((feature_agent,f_vel,fsp),0)
        #(32)+(16)+(48)=(96)
        assert input_x.shape[0]==96
        rt = torch.sigmoid(torch.sum(torch.mul(input_x,self.weight_ir))+self.bias_ir+torch.sum(torch.mul(hidden,self.weight_hr))+self.bias_hr)
        zt = torch.sigmoid(torch.sum(torch.mul(input_x,self.weight_iz))+self.bias_iz+torch.sum(torch.mul(hidden,self.weight_hz))+self.bias_hz)
        nt = torch.tanh(torch.sum(torch.mul(input_x,self.weight_in))+self.bias_in+rt*(torch.sum(torch.mul(hidden,self.weight_hn))+self.bias_hn))
        ht = (1-zt)*nt + zt*hidden
        return ht


class GRULayer(nn.Module):
    def __init__(self, cell, numbers_layers,*cell_args):
        super(GRULayer, self).__init__()
        self.cell = cell(*cell_args)

    #@jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        inputs = input.unbind(0)
        outputs = []
        for i in range(len(inputs)):
            state = self.cell(inputs[i], state)
            outputs += [state]
        return torch.stack(outputs), state

class SCF_GRULayer(nn.Module):
    def __init__(self, cell, batch_size, nums_sample,numbers_layers,*cell_args):
        super(SCF_GRULayer, self).__init__()
        self.cell = cell(*cell_args)
        self.cell.to(DEVICE)
        self.K = nums_sample
        self.numbers_layers = numbers_layers
        self.batch_size = batch_size

    #@jit.script_method
    def forward(self,path,f_vel,f_img):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        '''
        path:tensor(40,batch_size*n,2)
        f_vel:tensor(40,batch_size*n,16)
        f_img:tensor(batch_size,32,80,80)
        return output_x(40,batch_size*n,48),hidden_n(batch_size*n,48)
        '''
        assert path.shape[0]==40
        nums_agent = int(path.shape[1]/self.batch_size)
        outputs = []
        #state:tensor(n,hidden)
        state = torch.zeros((path.shape[1], self.cell.hidden_size), device=torch.device(DEVICE))
        # print(f_img.shape)
        # print("batch_size:{}".format(self.batch_size))
        for i in range(self.numbers_layers):
          new_state = state.clone()
          for k in range(self.batch_size):
              for j in range(nums_agent):
                # tensor(2)
                loc_agent = path[i][k*nums_agent+j]
                # tensor(nums_agent,2)
                loc_other = torch.zeros((nums_agent-1, 2), device=torch.device(DEVICE))
                loc_other_index = []
                index = 0
                for t in range(nums_agent):
                  if t != j:
                    loc_other[index] = path[i][k*nums_agent+t]
                    loc_other_index.append(t)
                    index += 1
                new_state[k*nums_agent+j] = self.cell(loc_agent, loc_other,
                                                      loc_other_index, f_img[k],
                                                      f_vel[i][k*nums_agent+j], state[i:i+nums_agent], state[k*nums_agent+j])
          state = new_state.clone()
          outputs += [state]
        return torch.stack(outputs), state

class GRU(nn.Module):
    def __init__(self, input_size,hidden_size,nums_layers):
        super(GRU, self).__init__()
        self.layer = GRULayer(GRUCell,nums_layers,input_size,hidden_size)

    #@jit.script_method
    def forward(self, input, state):
        return self.layer.forward(input,state)

class SCF_GRU(nn.Module):
    def __init__(self, batch_size, nums_sample,input_size,hidden_size,nums_layers,radius_range,social_pooling_size,device):
        global DEVICE
        super(SCF_GRU, self).__init__()
        DEVICE=device
        self.layer = SCF_GRULayer(SCF_GRUCell, batch_size, nums_sample,nums_layers,input_size,hidden_size,radius_range,social_pooling_size)
        self.layer.to(DEVICE)
    #@jit.script_method
    def forward(self, Y_path,Y_fv,feature_map):
      '''
      Y_path:Tensor (40,n,2)
      Y_fv: Tensor (40,n,16)
      feature_map: Tensor (batch_size,32,80,80)
      '''
      return self.layer.forward(Y_path,Y_fv,feature_map)

