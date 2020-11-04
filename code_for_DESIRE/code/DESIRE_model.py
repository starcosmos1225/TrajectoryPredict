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
#@cuda.jit

class HalfExp(nn.Module):
    def __init__(self):
        super(HalfExp, self).__init__()
 
    def forward(self, x):
        x = 0.5 * torch.exp(x)
        return x

class CVAEModel(nn.Module):
  def __init__(self,sample_number = 4,hz=10,device='cpu',batch_size=1,nums_iteration=1):
    super(CVAEModel, self).__init__()
    self.K = sample_number
    self.device = device
    self.batch_size = batch_size
    self.build()               
     

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
    #print(trajectory_data_x)
    #t=input()
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
    #print(Hx.shape)
    #t=input()
    # (batch_size*n,16,20)->(20,batch_size*n,16)
    Hx = Hx.permute(2, 0, 1)
    # (20,batch_size*n,16)->(20,batch_size*n,48)
    Hx,h_n_x = self.encoder1_gru(Hx)
    # (batch_size*n,48)
    new_Hx = Hx[-1]
    # Hy :(n,2,40)->(n,16,40)
    Hy = self.rnn_encoder2(trajectory_data_y)
    Hy = Hy.permute(2,0,1)
    # (40,batch_size*n,16)->(40,batch_size*n,48)
    Hy,h_n_y = self.encoder2_gru(Hy)
    # (batch_size*n,48)
    new_Hy = Hy[-1]
    # Hxy :(batch_size*n,48)+(batch_size*n,48)->(batch_size*n,96)
    Hxy = torch.cat((new_Hx,new_Hy),1)
    # CVAE
    # Hc:(batch_size*n,96)->(batch_size*n,48)
    Hc = self.fc1(Hxy)
    size_n = Hc.shape
    # H_miu:(batch_size*n,48)->(batch_size*n,48)
    H_miu = self.fc2(Hc)
    # H_delta:(batch_size*n,48)->(batch_size*n,48)
    H_delta = self.fc3(Hc)
    # sample k paths
    Y_path = []#torch.zeros((self.K,trajectory_data_y.shape[2],
                #          trajectory_data_y.shape[0], trajectory_data_y.shape[1]), device=torch.device(self.device))
    #Z K*(n,48)
    # Z = []
    # record each sample's score
    for i in range(self.K):
      #(batch_size*n,48)
      #print("K:{}".format(i))
      normalize = torch.randn((size_n), device=torch.device(self.device)).detach()
      #z_i:(batch_size*n,48)
      z_i = H_delta.mul(normalize)+H_miu
      # Z.append(z_i)
      #beta_z:(batch_size*n,48)
      beta_z = self.fc4(z_i)
      # mask
      # xz_i:(batch_size*n,48)
      xz_i = new_Hx.mul(beta_z)
      # padding 0 for gru input:(batch_size*n,48)->(40,batch_size*n,48)
      xz = []
      for j in range(sequence_y):
        if j==0:
          xz.append(xz_i)
        else:
          xz.append(torch.zeros_like(xz_i))
      xz = torch.stack(xz)
      # xz = []torch.zeros((sequence_y,size_n[0],size_n[1]), device=torch.device(self.device))
      # xz[0] = xz_i
      # reconstruction
      Hxz_i,h_x_xz = self.sample_reconstruction(xz)
      #h_size = Hxz_i.shape[0]
      # Y_i is the initial predict path:(40,batch_size*n,2)
      Y_i = self.fc5(Hxz_i)
      #record the predict path
      Y_path.append(Y_i)
    return new_Hx,torch.stack(Y_path), H_miu,H_delta


  def compute_dist_loss(self, Y_i, Y):
    '''
    Y_i:predict path:(K,40,batch_size*n,2)
    Y  :ground truth: (batch_size*n,2,40)
    '''
    loss_sum = torch.zeros(1).to(self.device)
    # (batch_size*n,2,40)->(40,batch_size*n,2)
    Y_gt= Y.permute(2,0,1)
    for i in range(self.K):
      #print((Y_i[i]-Y_gt).norm().shape)
      #t=input()
      loss_sum += (Y_i[i]-Y_gt).norm()
    return loss_sum/self.K/Y.shape[0]

  def compute_kld(self,miu,sigma):
    '''
    miu:(batch_size*n,48)
    sigma:(batch_size*n,48)
    return :the loss of KLD:-0.5*(1+log(sigma*sigma)-sigma*sigma-miu*miu)
    '''
    sigma2 = torch.square(sigma)
    return torch.mean(-0.5*(1+torch.log(sigma2+1e-10)-sigma2-torch.square(miu)))

  def train(self, trajectory_data_x,trajectory_data_y):
    '''
    input:
    trajectory_data_x: a tensor with shape (batch_size,10,2,20)
    trajectory_data_y: a tensor with shape (batch_size,10,2,40)
    '''
    # print("begin train")
    # predict_path:(K,40, batch_size*n, 2)
    # delta_y:(K,40, batch_size*n, 2)
    # scores: (K,batch_size*n, 1)
    # miu: (batch_size*n, 48)
    # sigma: (batch_size*n,48)
    trajectory_data_x = trajectory_data_x.view(-1,trajectory_data_x.shape[2],trajectory_data_x.shape[3])
    trajectory_data_y = trajectory_data_y.view(-1,trajectory_data_y.shape[2],trajectory_data_y.shape[3])
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    start = time.time()
    hx,predict_path,miu,sigma = self.forward(trajectory_data_x,trajectory_data_y)
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    end = time.time()
    print("the cvae forward time:{}".format(end-start))
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    start = time.time()
    loss_distance = self.compute_dist_loss(predict_path,trajectory_data_y)
    loss_kld = self.compute_kld(miu,sigma)
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    end = time.time()
    loss = loss_distance+loss_kld
    print("the cvae loss compute time:{}".format(end-start))
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
    self.social_pooling_size=(6,6)
    self.radius_range = (0.5,4)
    self.build()               
     

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
    bn = hx.shape[0]
    hx = hx.repeat((self.K,1))
    
    sequence_y = y_path.shape[1]
    
    n_agents = int(bn/self.batch_size)
    # cnn feature map （batch_size,4,160,160）->(batch_size,32,80,80)
    feature_map = self.cnn_map(image_data)
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    start = time.time()
    # Y_velocity is velocity tensor(k,40,batch_size*n,2)
    Y_velocity = self.compute_vel(y_path, current_location)
    
    #(40,batch_size*n,2)->(40,batch_size*n,16)
    Y_fv = self.fc_vel(Y_velocity)
    # tensor(40,batch_size*n,48)
    scf_gru_hidden = []
    # (batch_size*n,48)
    for j in range(self.iteration):
      for it in range(sequence_y):
        lhalf_i = None
        sps = None
        for batch in range(self.batch_size):
          if batch==0:
            #(k,n,48) (k,n,36*48)
            lhalf_i,sps = self.scf(y_path[:,:,batch:batch+n_agents,:],Y_fv[:,:,batch:batch+n_agents,:],
                                                              feature_map[batch],hx[batch:batch+n_agents],n_agents,it)
          else:
            #(k,n,48) (k,n,36*48)
            tmp_l,tmp_s = self.scf(y_path[:,:,batch:batch+n_agents,:],Y_fv[:,:,batch:batch+n_agents,:],
                                                              feature_map[batch],hx[batch:batch+n_agents],n_agents,it)
            lhalf_i = torch.cat((lhalf_i,tmp_l),1)
            sps = torch.cat((sps,tmp_s),1)
          #print(lhalf_i.shape)
          #t=input()                     
        # (k,batch_size*n,36*48)->(k,batch_size*n,48)                              
        rhalf_i = self.fc_scf(sps)
        #(k,batch_size*n,48)+(k,batch_size*n,48)=(k,batch_size*n,96)->(k*batch_size*n,96)
        x_i = torch.cat((lhalf_i,rhalf_i),2).view(self.K*self.batch_size*n_agents,-1)
        #(k*batch_size*n,96)->(k*batch_size*n,48)
        hx = self.GRU_cell(x_i, hx)
        #print(hx.shape)100
        #t=input()
        if j==0:
          scf_gru_hidden.append(hx)
        else:
          scf_gru_hidden[it] = hx
      #(k*batch_size*n,48)->(k*batch_size*n,80)->(K,batch_size*n,2,40)->(K,40,batch_size*n,2)
      deltaY = self.fc_dy(hx).view(self.K,bn,2,-1).permute(0,3, 1, 2).contiguous()
      y_path = (y_path+deltaY).detach()
    #40 list of (k*batch_size*n,48)->(40,k*batch_size*n,48)
    scf_gru_hidden = torch.stack(scf_gru_hidden)
    
    #delta_Y_list.append(deltaY)
    #(40,k*batch_size*n,48)->(40,k*batch_size*n,1)->(k*batch_size*n,1)->(k,batch_size*n,1)
    score = torch.sum(self.fc_score(scf_gru_hidden), dim=0).view(self.K,self.batch_size*n_agents,1)
    #scores.append(score)
    return deltaY, score
  def compute_dist(self,loc_a,loc_b):
    '''
    loc_a:tensor(k,2)
    loc_b:tensor(k,2)
    return: distance (a-b) tensor(k) float
    '''
    return torch.norm(loc_a-loc_b,dim=1)
  
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
    costheta = c[:,0]/dist
    for i in range(l):    
      if c[i,1]<0:
        costheta[i] = 2*math.pi - costheta[i].acos()
      else:
        costheta[i] = costheta[i].acos()
    return costheta


  def scf(self,path,Y_fv,feature_map,hidden,nums_agent,index):
    '''
    path:(k,40,n,2)
    Y_fv:(k,40,n,16)
    feature_map:(32,80,80)
    hidden:(n,48)
    return:(k,n,48) (k,n,36*48)
    '''
    #(32,80,80)->(K,32,80,80)
    f_map = feature_map.unsqueeze(dim=0).repeat((self.K,1,1,1))
    H = feature_map.shape[1]
    hx = []#torch.zeros((nums_agent,48),device=torch.device(self.device))
    sps = []#torch.zeros((nums_agent,36*48),device=torch.device(self.device))
    #print(k)
    #print(type(k))
    #t=input()
    #print("begin scf")
    k_list = torch.arange(0,self.K,1).long()
    for j in range(nums_agent):
      #print("index:{} j:{}".format(index,j))
      # tensor(k,2)
      loc_agent = path[:,index,j,:]
      # tensor(nums_agent-1,k,2)
      loc_others = []#torch.zeros((nums_agent-1, 2), device=torch.device(self.device))
      loc_other_index = []
      count = 0
      for t in range(nums_agent):
        if t != j:
          # (k,2)
          loc_others.append(path[:,index,t])
          loc_other_index.append(t)
          count += 1
      #list(k,2)->(n-1,k,2)->(k,n-1,2)
      loc_others = torch.stack(loc_others).permute(1,0,2)
      u = int(H/2)-loc_agent[:,1].long()
      v = int(H/2)-loc_agent[:,0].long()
      # feature_agent:(k,32)
      feature_agent = f_map[k_list,:, u, v]
      # sp: tensor(K,6,6,48)
      sp = torch.zeros((self.K,self.social_pooling_size[0]*self.social_pooling_size[1],hidden.shape[1]), device=torch.device(self.device))
      # sp_c: count the numbers in (K,6,6)
      sp_c = torch.zeros((self.K,self.social_pooling_size[0]*self.social_pooling_size[1]), device=torch.device(self.device),dtype=torch.int32).detach()
      sp_one = torch.ones_like(sp_c)
      #print("after spc")
      for i in range(loc_others.shape[1]):
        # loc:tensor(k,2)
        loc = loc_others[:,i,:]
        # dist:tensor(k)
        dist = self.compute_dist(loc, loc_agent).detach()
        dist_index = torch.where((dist<=self.radius_range[1])&(dist>=self.radius_range[0]))[0]
        if dist_index.shape[0]!=0:
          theta = self.compute_theta(loc_agent[dist_index,:], loc[dist_index,:]).detach()
          u = ((dist[dist_index]-self.radius_range[0])/self.radius_step).int()
          v = (theta/self.theta_step).int()
          #index: (k)
          loc_index = u*self.social_pooling_size[1]+v
          sp[dist_index,loc_index] = hidden[loc_other_index[i]]
          sp_c[dist_index,loc_index] += 1
      sp_c = torch.where(sp_c == 0, sp_one, sp_c)
      sp_c = sp_c.unsqueeze(dim=-1).repeat([1,1,hidden.shape[1]])
      sp = sp/sp_c
      #(k,6*6,48)->(k,6*6*48)
      #t=input()
      sp = sp.view((self.K,self.social_pooling_size[0]*self.social_pooling_size[1]*hidden.shape[1]))
      sps.append(sp)
      #(k,32)+(k,16)
      #print(feature_agent.shape)
      #print(Y_fv[:,index,j,:].shape)
      #t=input()
      #print(Y_fv.shape)
      input_x = torch.cat((feature_agent,Y_fv[:,index,j,:]),1)
      hx.append(input_x)
    return torch.stack(hx).permute(1,0,2),torch.stack(sps).permute(1,0,2)

  def compute_vel(self,path,current_location):
    '''
    path:tensor(k,40,batch_size*n,2)
    current_location:the current location for agents.Tensor with size(batch_size*n,2)
    '''
    sequence = path.shape[1]
    vel = torch.zeros(path.shape).to(self.device).detach()
    # (batch_size*n,2)->(K,batch_size*n,2)
    loc = current_location.unsqueeze(dim=0).repeat((self.K,1,1))
    for j in range(sequence):
      if j == 0:
        vel[:,j,:,:] = (path[:,j,:,:]-loc)
      else:
        vel[:,j,:,:] = (path[:,j,:,:]-path[:,j-1,:,:])
    vel = vel*self.hz
    return vel

  def compute_cross_entropy(self,oldY,newY,Y):
    '''
    oldY:(K,40,batch_size*n,2)
    newY:(K,40,batch_size*n,2)
    Y:(batch_size*n,2,40)
    '''
    #(n,2,40)->(40,n,2)->(K,40,n,2)
    Y_resize = Y.permute(2,0,1).unsqueeze(dim=0).repeat((self.K,1,1,1))
    #loss = torch.zeros(1).to(self.device)
    #for i in range(self.K):
      #dist (40, n)
    old_d = torch.max(torch.abs(oldY-Y_resize),dim=2).values
    new_d = torch.max(torch.abs(newY-Y_resize),dim=2).values
    # P, Q (40, n)
    P = func.softmax(old_d,dim=0)
    Q = func.softmax(new_d,dim=0)
    Hpq = torch.sum(-P*torch.log(Q))
    return Hpq

  def compute_regression(self,Y_i,Y):
    '''
    Y_i:predict path:K list with (k,40,batch_size*n,2)
    Y  :ground truth: (40,batch_size*n,2)
    '''
    Y_ = Y.unsqueeze(dim=0).repeat((self.K,1,1,1))
    return (Y_i-Y_).norm()/self.K/Y.shape[1]

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
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    start = time.time()
    delta_y,scores = self.forward(hx,current_location,y_path,feature_image)
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    end = time.time()
    print("the refine mode forward time:{}".format(end-start))
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    start = time.time()
    new_predict_path = y_path+delta_y
    loss_ce = self.compute_cross_entropy(y_path,new_predict_path,trajectory_data_y)
    loss_regression = self.compute_regression(new_predict_path,trajectory_data_y.permute(2,0,1))
    loss = loss_ce+loss_regression
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    end = time.time()
    print("the loss compute time:{}".format(end-start))
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
