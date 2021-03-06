#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable 
import torch.nn.functional as func 
import numpy as np

import threading
#from SCF_GRU import SCF_GRU,GRU_TEST
import time
import math
test1 = 0.0
#@cuda.jit
class HalfExp(nn.Module):
    def __init__(self):
        super(HalfExp, self).__init__()
 
    def forward(self, x):
        x_ = 0.5 * torch.exp(x)
        return x_

class CVAEModel(nn.Module):
  def __init__(self,sample_number = 4,hz=10,device='cpu',batch_size=1,nums_iteration=1):
    super(CVAEModel, self).__init__()
    self.K = sample_number
    self.device = device
    self.batch_size = batch_size
    self.build()               
     
  def init(self,sample_number = 4,hz=10,device='cpu',batch_size=1,nums_iteration=1):
    self.K = sample_number
    self.device = device
    self.batch_size = batch_size

  def forward(self, trajectory_data_x,trajectory_data_y):
    
    '''
    input:
    trajectory_data_x: a tensor with shape (batch_size*n,2,20) 
    trajectory_data_y: a tensor with shape (batch_size*n,2,40) 
    return:
      hx: the feature X
      Y_path:the K paths with cell(K,40, batch_size*n, 2)
      H_miu: (batch_size*n, 48)
      H_delta: (batch_size*n, 48)
    '''
    #global test1
    sequence_x = trajectory_data_x.shape[2]
    sequence_y = trajectory_data_y.shape[2]
    bn = trajectory_data_x.shape[0]
    n_agents = int(bn/self.batch_size)
    #current_location = trajectory_data_x[:, :, -1].detach()
    # cnn feature map （batch_size,4,160,160）->(batch_size,32,80,80)
    #feature_map = self.cnn_map(image_data)
    # Encoder 1 and 2
    # Hx :(batch_size*n,2,20)->(batch_size*n,16,20)
    
    Hx = self.rnn_encoder1(trajectory_data_x)
    
    # (batch_size*n,16,20)->(20,batch_size*n,16)
    Hx = Hx.permute(2, 0, 1)
    # (20,batch_size*n,16)->(20,batch_size*n,48)
    Hx_1,h_n_x = self.encoder1_gru(Hx)
    
    # (batch_size*n,48)
    new_Hx = Hx_1[-1]
    # Hy :(n,2,40)->(n,16,40)
    Hy = self.rnn_encoder2(trajectory_data_y)
    Hy = Hy.permute(2,0,1)
    # (40,batch_size*n,16)->(40,batch_size*n,48)
    Hy_1,h_n_y = self.encoder2_gru(Hy)
    # (batch_size*n,48)
    new_Hy = Hy_1[-1]
    # Hxy :(batch_size*n,48)+(batch_size*n,48)->(batch_size*n,96)
    Hxy = torch.cat((new_Hx,new_Hy),1)
    # CVAE
    # Hc:(batch_size*n,96)->(batch_size*n,48)
    Hc = self.fc1(Hxy)
    
    #Hc = torch.randn((bn,48))
    size_n = Hc.shape
    # H_miu:(batch_size*n,48)->(batch_size*n,48)
    H_miu = self.fc2(Hc)
    # H_delta:(batch_size*n,48)->(batch_size*n,48)
    H_delta = self.fc3(Hc)
    
    # sample k paths
    normalize = torch.randn((self.K,size_n[0],size_n[1]), device=torch.device(self.device)).detach()
    mul_H = H_delta.unsqueeze(dim=0).repeat((self.K,1,1)).to(self.device)
    mul_H_miu = H_miu.unsqueeze(dim=0).repeat((self.K,1,1)).to(self.device)
    #z_i:(K,batch_size*n,48)*(K,batch_size*n,48) + (K,batch_size*n,48) = (K,batch_size*n,48)
    z_i = mul_H.mul(normalize)+mul_H_miu
    #beta_z:(K,batch_size*n,48)
    beta_z = self.fc4(z_i)
    # mask
    mul_new_Hx = new_Hx.unsqueeze(dim=0).repeat((self.K,1,1)).to(self.device)
    # xz_i:(K,batch_size*n,48)->(K*batch_size*n,48)
    xz_i = mul_new_Hx.mul(beta_z).view(-1,48)
    # padding 0 for gru input:(batch_size*n,48)->(40,batch_size*n,48)
    xz = []
    for j in range(sequence_y):
      if j==0:
        xz.append(xz_i)
      else:
        xz.append(torch.zeros_like(xz_i))
    # xz:(40,K*batch_size*n,48)
    xz = torch.stack(xz)
    # reconstruction
    Hxz_i,h_x_xz = self.sample_reconstruction(xz)
    # Y_i is the initial predict path:(40,K*batch_size*n,2)->(K,40, batch_size*n, 2)
    Y_i = self.fc5(Hxz_i).view(40,self.K,-1,2).permute(1,0,2,3)
    
    return new_Hx,Y_i, H_miu,H_delta


  def compute_dist_loss(self, Y_i, Y):
    '''
    Y_i:predict path:(K,40,batch_size*n,2)
    Y  :ground truth: (batch_size*n,2,40)
    '''
    #loss_sum = torch.zeros(1).to(self.device)
    # (batch_size*n,2,40)->(40,batch_size*n,2)
    Y_gt= Y.permute(2,0,1).unsqueeze(dim=0).repeat((self.K,1,1,1)).to(self.device)

    loss_sum = (Y_i-Y_gt).norm(dim=3).sum()
    
    return loss_sum/self.K/10

  def compute_kld(self,miu,sigma):
    '''
    miu:(batch_size*n,48)
    sigma:(batch_size*n,48)
    return :the loss of KLD:-0.5*(1+log(sigma*sigma)-sigma*sigma-miu*miu)
    '''
    sigma2 = torch.square(sigma)
    return (-0.5*(1+torch.log(sigma2+1e-10)-sigma2-torch.square(miu))).sum()/10

  def train(self, trajectory_data_x,trajectory_data_y):
    '''
    input:
    trajectory_data_x: a tensor with shape (batch_size,10,2,20)
    trajectory_data_y: a tensor with shape (batch_size,10,2,40)
    '''


    trajectory_data_x = trajectory_data_x.view(-1,trajectory_data_x.shape[2],trajectory_data_x.shape[3])
    trajectory_data_y = trajectory_data_y.view(-1,trajectory_data_y.shape[2],trajectory_data_y.shape[3])

    hx,predict_path,miu,sigma = self.forward(trajectory_data_x,trajectory_data_y)

    loss_distance = self.compute_dist_loss(predict_path,trajectory_data_y)
    loss_kld = self.compute_kld(miu,sigma)
    
    loss = loss_distance+loss_kld
    return hx,predict_path,loss

  def build(self):
    self.rnn_encoder1 = nn.Sequential(nn.Conv1d(2,16,kernel_size=3,padding=1),
                                      nn.ReLU())
    self.encoder1_gru = nn.GRU(16,48,1)
    self.rnn_encoder2 = nn.Sequential(nn.Conv1d(2,16,kernel_size=1),
                                      nn.ReLU())
                                  
    self.encoder2_gru = nn.GRU(16,48,1)
    #must concat first: (48,48)->96
    self.fc1 = nn.Sequential(nn.Linear(96,48),nn.ReLU())
    self.fc2 = nn.Linear(48,48)
    self.fc3 = nn.Sequential(nn.Linear(48,48),HalfExp())
    self.fc4 = nn.Sequential(nn.Linear(48,48),nn.Softmax(dim=1))
    # multiplication
    self.sample_reconstruction = nn.GRU(48,48,1)
    self.fc5 = nn.Linear(48,2)

class RefineModel(nn.Module):
  def __init__(self,sample_number = 4,hz=10,device='cpu',batch_size=1,nums_iteration=1):
    super(RefineModel, self).__init__()
    self.K = sample_number
    self.hz = hz
    self.device = device
    self.batch_size = batch_size
    self.iteration = nums_iteration
    self.social_pooling_size=torch.tensor([6,6],device=torch.device(self.device))
    self.radius_range = torch.tensor([0.5,4.0],device=torch.device(self.device))
    self.radius_step = (self.radius_range[1]-self.radius_range[0])/self.social_pooling_size[0]
    self.theta_step = torch.tensor([2*math.pi],device=torch.device(self.device))/self.social_pooling_size[1]
    self.build()               

  def init(self,sample_number = 4,hz=10,device='cpu',batch_size=1,nums_iteration=1):
    self.K = sample_number
    self.hz = hz
    self.device = device
    self.batch_size = batch_size
    self.iteration = nums_iteration
    self.social_pooling_size=torch.tensor([6,6],device=torch.device(self.device))
    self.radius_range = torch.tensor([0.5,4.0],device=torch.device(self.device))
    self.radius_step = (self.radius_range[1]-self.radius_range[0])/self.social_pooling_size[0]
    self.theta_step = torch.tensor([2*math.pi],device=torch.device(self.device))/self.social_pooling_size[1]        

  def forward(self, hx,current_location,y_path,image_data):
    '''
    input:
    hx: (batch_size*n,48)
    current_location:(batch_size*n,2)
    y_path:(k,40,batch_size*n,2)
    image_data: a tensor with shape (batch_size,4,160,160)
    return:
      deltaY:the K delta path with cell(K,40, batch_size*n, 2)
      scores:the K paths' scores with cell(K,batch_size*n, 1)
    '''
    global test1
    bn = hx.shape[0]
    hx_ = hx.repeat((self.K,1)).to(self.device)
    
    sequence_y = y_path.shape[1]
    
    n_agents = int(bn/self.batch_size)
    # cnn feature map （batch_size,4,160,160）->(batch_size,32,80,80)
    feature_map = self.cnn_map(image_data)
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    start = time.time()
    # Y_velocity is velocity tensor(k,40,batch_size*n,2)
    Y_velocity = self.compute_vel(y_path, current_location).detach()
    
    #(40,batch_size*n,2)->(40,batch_size*n,16)
    Y_fv = self.fc_vel(Y_velocity)
    # tensor(40,batch_size*n,48)
    scf_gru_hidden = []
    # (batch_size*n,48)
    for j in range(self.iteration):
      for it in range(sequence_y):
        lhalf_i,sps = self.scf(y_path[:,it,:,:], Y_fv[:,it,:,:], feature_map, hx_, n_agents)                 
        # (k,batch_size*n,36*48)->(k,batch_size*n,48)                              
        rhalf_i = self.fc_scf(sps)
        #(k,batch_size*n,48)+(k,batch_size*n,48)=(k,batch_size*n,96)->(k*batch_size*n,96)
        x_i = torch.cat((lhalf_i,rhalf_i),2).view(self.K*self.batch_size*n_agents,-1)
        #(k*batch_size*n,96)->(k*batch_size*n,48)
        hx_ = self.GRU_cell(x_i, hx_)
        #print(hx.shape)100
        #t=input()
        if j==0:
          scf_gru_hidden.append(hx_)
        else:
          scf_gru_hidden[it] = hx_
      #(k*batch_size*n,48)->(k*batch_size*n,80)->(K,batch_size*n,2,40)->(K,40,batch_size*n,2)
      deltaY = self.fc_dy(hx_).view(self.K,bn,2,-1).permute(0,3, 1, 2).contiguous()
      #y_path = (y_path+deltaY).detach()
    #40 list of (k*batch_size*n,48)->(40,k*batch_size*n,48)
    scf_gru_hidden = torch.stack(scf_gru_hidden)
    
    #delta_Y_list.append(deltaY)
    #(40,k*batch_size*n,48)->(40,k*batch_size*n,1)->(k*batch_size*n,1)->(k,batch_size*n,1)
    score = torch.sum(self.fc_score(scf_gru_hidden), dim=0).view(self.K,self.batch_size*n_agents,1)
    #scores.append(score)
    return deltaY, score
  def compute_dist(self,loc_a,loc_b):
    '''
    loc_a:tensor(k,batch_size,2)
    loc_b:tensor(k,batch_size,2)
    return: distance (a-b) tensor(k,batch_size) float
    '''
    return torch.norm(loc_a-loc_b,dim=2)
  
  def compute_theta(self,loc_a,loc_b):
    '''
    loc_a:tensor(k,2)
    loc_b:tensor(k,2)
    return: angle (b-a) tensor(k,1) float(0~2pi)
    '''
    l = loc_a.shape[0]
    c = loc_b-loc_a
    dist = torch.norm(c,dim=1)
    min_dist = torch.ones_like(dist)*1e-10
    dist = torch.where(dist<1e-10,min_dist,dist)
    costheta = (c[:,0]/dist).acos()
    neg_index = torch.where(c[:,1]<-1e-2)
    #print(costheta)
    costheta[neg_index]= 2*math.pi - costheta[neg_index]
    #print(costheta)
    # #t=input()
    # for i in range(l):    
    #   if c[i,1]<0:
    #     costheta[i] = 2*math.pi - costheta[i].acos()
    #     if abs(2*math.pi-costheta[i])<1e-4:
    #       print("change to")
    #       costheta[i]=0
    #   else:
    #     costheta[i] = costheta[i].acos()
    return costheta


  def scf(self,path,Y_fv,feature_map,hidden,nums_agent):
    '''
    path:(k,batch_size*n,2)
    Y_fv:(k,batch_size*n,16)
    feature_map:(batch_size,32,80,80)
    hidden:(batch_size*n,48)
    return:(k,batch_size*n,48) (k,batch_size*n,36*48)
    '''
    #(batch_size,32,80,80)->(K*batch_size,32,80,80)
    f_map = feature_map.repeat((self.K,1,1,1)).to(self.device)
    H = feature_map.shape[2]
    W = feature_map.shape[3]
    hx = []
    sps = []
    kb_list = torch.arange(0,self.K*self.batch_size,1)
    for j in range(nums_agent):
      # tensor(k,batch_size,2)
      loc_agent = path[:,j::nums_agent,:]
      # tensor(nums_agent-1,k,batch_size,2)
      loc_others = []
      loc_other_index = []
      count = 0
      for t in range(nums_agent):
        if t != j:
          # (k,2)
          loc_others.append(path[:,t::nums_agent,:].detach())
          loc_other_index.append(t)
          count += 1
      #list(k,batch_size,2)->(n-1,k,batch_size,2)
      loc_others = torch.stack(loc_others).detach()
      u = int(H/2)-loc_agent[:,:,1].reshape(self.K*self.batch_size).long()
      v = loc_agent[:,:,0].reshape(self.K*self.batch_size).long()
      torch.clamp(u, min=0, max=H-1, out=u)
      torch.clamp(v, min=0, max=W-1, out=v)
      #print(u)
      #print(v)
      #t=input()
      HH = torch.tensor(H,device=torch.device(self.device))
      WW = torch.tensor(W,device=torch.device(self.device))
      #index = torch.where((u>=0)&(u<HH)&(v>=0)&(v<WW))
      ##kb_list = kb_list[index]
      #u = u[index]
      #v = v[index]
      # feature_agent:(k*batch_size,32)
      feature_agent = f_map[kb_list,:,u, v].view(self.K,self.batch_size,32)
      # sp: tensor(K,batch_size,6*6,48)
      sp = torch.zeros((self.K,self.batch_size,int(self.social_pooling_size[0].item()*self.social_pooling_size[1].item()),hidden.shape[1]), device=torch.device(self.device))
      # sp_c: count the numbers in (K,batch_size,6*6)
      sp_c = torch.zeros((self.K,self.batch_size,int(self.social_pooling_size[0].item()*self.social_pooling_size[1].item())), device=torch.device(self.device),dtype=torch.int32).detach()
      sp_one = torch.ones_like(sp_c)
      for i in range(loc_others.shape[0]):
        # loc:tensor(k,batch_size,2)
        loc = loc_others[i,:,:,:]
        # dist:tensor(k,batch_size)
        dist = self.compute_dist(loc, loc_agent).detach()
        #list of (index_shape) ,[dim0,dim1]
        dist_index = torch.where((dist<=self.radius_range[1])&(dist>=self.radius_range[0]))
        if dist_index[0].shape[0]!=0:
          theta = self.compute_theta(loc_agent[dist_index[0],dist_index[1],:], loc[dist_index[0],dist_index[1],:]).detach()
          u = ((dist[dist_index[0],dist_index[1]]-self.radius_range[0])/self.radius_step).long()
          v = (theta/self.theta_step).long()
          torch.clamp(u, min=0, max=self.social_pooling_size[0]-1, out=u)
          torch.clamp(v, min=0, max=self.social_pooling_size[1]-1, out=v)
          #index: (index_shape)
          loc_index = u*self.social_pooling_size[1]+v
          hidden_index = dist_index[1]*nums_agent + loc_other_index[i]
          sp[dist_index[0],dist_index[1],loc_index] = sp[dist_index[0],dist_index[1],loc_index]+hidden[hidden_index]
          sp_c[dist_index[0],dist_index[1],loc_index] += 1
      sp_c = torch.where(sp_c == 0, sp_one, sp_c)
      sp_c = sp_c.unsqueeze(dim=-1).repeat([1,1,1,hidden.shape[1]]).to(self.device).detach()
      sp = sp/sp_c
      #(k,batch_size,6*6,48)->(k,batch_size,6*6*48)
      sp = sp.view((self.K,self.batch_size,self.social_pooling_size[0]*self.social_pooling_size[1]*hidden.shape[1]))
      sps.append(sp)
      #(k,batch_size,32)+(k,batch_size,16)->(k,batch_size,48)
      input_x = torch.cat((feature_agent,Y_fv[:,j::nums_agent,:]),2)
      hx.append(input_x)
    # hx:(n,k,batch_size,48)->(k,batch_size,n,48)->(k,batch_size*n,48)
    # sps:(n,k,batch_size,6*6*48)->(k,batch_size,n,6*6*48)->(k,batch_size*n,6*6*48) 
    return torch.stack(hx).permute(1,2,0,3).reshape(self.K,self.batch_size*nums_agent,48),\
            torch.stack(sps).permute(1,2,0,3).reshape(self.K,self.batch_size*nums_agent,6*6*48)

  def compute_vel(self,path,current_location):
    '''
    path:tensor(k, 40, batch_size*n, 2)
    current_location:the current location for agents.Tensor with size(batch_size*n, 2)
    '''
    sequence = path.shape[1]
    vel = torch.zeros(path.shape).to(self.device).detach()
    # (batch_size*n,2)->(K,batch_size*n,2)
    loc = current_location.unsqueeze(dim=0).repeat((self.K,1,1)).to(self.device)
    for j in range(sequence):
      if j == 0:
        vel[:,j,:,:] = (path[:,j,:,:]-loc)
      else:
        vel[:,j,:,:] = (path[:,j,:,:]-path[:,j-1,:,:])
    vel = vel*self.hz
    return vel

  def compute_cross_entropy(self,newY,Y):
    '''
    oldY:(K, 40, batch_size*n, 2)
    newY:(K, 40, batch_size*n, 2)
    Y:(batch_size*n,2,40)
    '''
    #(n,2,40)->(40,n,2)->(K,40,n,2)
    Y_resize = Y.permute(2,0,1).unsqueeze(dim=0).repeat((self.K,1,1,1)).to(self.device)

    new_d = torch.max(torch.abs(newY-Y_resize),dim=3).values
  
    Q = func.softmax(new_d,dim=1)
    Hpq = (-1.0/40.0*torch.log(Q)).sum()/self.K/10
    return Hpq

  def compute_regression(self,Y_i,Y):
    '''
    Y_i:predict path:K list with (k,40,batch_size*n,2)
    Y  :ground truth: (40,batch_size*n,2)
    '''
    Y_ = Y.unsqueeze(dim=0).repeat((self.K,1,1,1)).to(self.device)
    loss = (Y_i-Y_).norm(dim=3).sum()
    return loss/self.K/10

  def train(self, hx,current_location,y_path,feature_image,trajectory_data_y):
    '''
    input:
    hx: (batch_size*n,48)
    current_location(batch_size*n,2)
    y_path:(k,40,batch_size*n,2)
    feature_image(batch_size,4,160,160)
    trajectory_data_y: a tensor with shape (batch_size,10,2,40)
    '''
    # delta_y:(K,40, batch_size*n, 2)
    # scores: (K,batch_size*n, 1)
    trajectory_data_y = trajectory_data_y.view(-1,trajectory_data_y.shape[2],trajectory_data_y.shape[3])
    
    init_y = y_path.detach()
    delta_y,scores = self.forward(hx,current_location,y_path,feature_image)
    
    new_predict_path = init_y+delta_y
    loss_ce = self.compute_cross_entropy(new_predict_path,trajectory_data_y)
    
    loss_regression = self.compute_regression(new_predict_path,trajectory_data_y.permute(2,0,1))
    loss = loss_ce+loss_regression

    return loss

  def build(self):
   
    # CNN for semantic map
    self.cnn_map = nn.Sequential(nn.Conv2d(4,16,kernel_size=(5,5),stride=2,padding=(2,2)),
                                 nn.ReLU(),
                                 nn.Conv2d(16,32,kernel_size=(5,5),stride=1,padding=(2,2)),
                                 nn.ReLU())
    # IOC scene context fusion
    # feature pooling: do pooling at (x,y) for map(H/2,W/2,32) get feature list[32]
    # get velocity(x,y)->fc (2)->(16). How to compute the velocity? V_Y can get by dist_i-dist_{i-1}/time_per_frame
    self.fc_vel = nn.Sequential(nn.Linear(2,16),nn.ReLU())
    # social pooling:for Y_{i},center is (x,y) get a circle(radius=4m)'s pooling feature(32). On log-polar grid ,we get 6*6 feature map
    # feature map(6*6*48)->fc->(48) get feature_sp
    # concat 32+16+48=96 scF
    # nn.GRU(96,48,40)->(1,40)
    self.GRU_cell = nn.GRUCell(96,48)
    self.fc_scf = nn.Sequential(nn.Linear(self.social_pooling_size[0]*self.social_pooling_size[1]*48,48),
                                 nn.ReLU())
    #self.decoder2 = SCF_GRU(self.batch_size, self.K,  96, 48, 40, radius_range=(0.5,4.0), social_pooling_size=(6,6),device=self.device)
    # for output(48)->(2,40)
    self.fc_score = nn.Linear(48,1)
    self.fc_dy =nn.Sequential(nn.Linear(48,80),nn.ReLU())

if __name__ == '__main__':
  pass
