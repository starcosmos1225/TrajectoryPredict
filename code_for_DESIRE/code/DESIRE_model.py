#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable 
import torch.nn.functional as func 
import numpy as np
from SCF_GRU import SCF_GRU

class HalfExp(nn.Module):
    def __init__(self):
        super(HalfExp, self).__init__()
 
    def forward(self, x):
        x = 0.5 * torch.exp(x)
        return x

class Model(nn.Module):
  def __init__(self,sample_number = 4,hz=10,device='cpu'): 
    super(Model, self).__init__()
    self.K = sample_number
    self.hz = hz
    self.device = device
    self.build()               
     

  def forward(self, trajectory_data_x,trajectory_data_y,image_data):
    '''
    input:
    trajectory_data_x: a tensor with shape (n,2,20) n is the numbers of object
    trajectory_data_y: a tensor with shape (n,2,40) n is the numbers of object
    image_data: a tensor with shape (batch_size,4,160,160)
    return:
      Y_path:the K paths with cell(K,40, n, 2)
      deltaY:the K delta path with cell(K,40, n, 2)
      scores:the K paths' scores with cell(K,n, 1)
    '''
    sequence_x = trajectory_data_x.shape[2]
    sequence_y = trajectory_data_y.shape[2]
    current_location = trajectory_data_x[:,:,-1].detach()
    # cnn feature map （batch_size,4,160,160）->(batch_size,32,80,80)
    feature_map = self.cnn_map(image_data)
    # Encoder 1 and 2
    # Hx :(n,2,20)->(n,16,20)
    Hx = self.rnn_encoder1(trajectory_data_x)
    # (n,16,20)->(20,n,16)
    Hx = Hx.permute(2,0,1)
    # (20,n,16)->(20,n,48)
    Hx,h_n_x = self.encoder1_gru(Hx)
    new_Hx = Hx[-1]
    # Hy :(n,2,40)->(n,16,40)
    Hy = self.rnn_encoder2(trajectory_data_y)
    Hy = Hy.permute(2,0,1)
    # (40,n,16)->(40,n,48)
    Hy,h_n_y = self.encoder2_gru(Hy)
    new_Hy = Hy[-1]
    # Hxy :(n,48)+(n,48)->(n,96)
    Hxy = torch.cat((new_Hx,new_Hy),1)
    # CVAE
    # Hc:(n,96)->(n,48)
    Hc = self.fc1(Hxy)
    size_n = Hc.shape
    # H_miu:(n,48)->(n,48)
    H_miu = self.fc2(Hc)
    # H_delta:(n,48)->(n,48)
    H_delta = self.fc3(Hc)
    # sample k paths
    Y_path = torch.zeros((self.K,trajectory_data_y.shape[2],trajectory_data_y.shape[0],trajectory_data_y.shape[1])).to(self.device)
    #Z K*(n,48)
    # Z = []
    # record each sample's score
    scores = torch.zeros((self.K,trajectory_data_y.shape[0],1)).to(self.device)
    delta_Y_list = torch.zeros(((self.K,trajectory_data_y.shape[2],trajectory_data_y.shape[0],trajectory_data_y.shape[1]))).to(self.device)
    for i in range(self.K):
      #(n,48)
      normalize = torch.randn(size_n).to(self.device)
      #z_i:(n,48)
      z_i = H_delta.mul(normalize)+H_miu
      # Z.append(z_i)
      #beta_z:(n,48)
      beta_z = self.fc4(z_i)
      # mask
      # xz_i:(n,48)
      xz_i = new_Hx.mul(beta_z)
      # padding 0 for gru input:(n,48)->(40,n,48)
      xz = torch.zeros((sequence_y,size_n[0],size_n[1])).to(self.device)
      xz[0] = xz_i
      # reconstruction
      Hxz_i,h_x_xz = self.sample_reconstruction(xz)
      #h_size = Hxz_i.shape[0]
      # Y_i is the initial predict path:(40,n,2)
      Y_i = self.fc5(Hxz_i)
      #record the predict path
      Y_path[i] = Y_i
      # Y_velocity is velocity tensor(40,n,2)
      Y_velocity = self.compute_vel(Y_i,current_location)
      #print(Y_velocity.shape)
      #(40,n,2)->(40,n,16)
      Y_fv = self.fc_vel(Y_velocity)
      #print(Y_fv.shape)
      #t=input('0')
      # hidden:tensor(40,n,48) state:tensor(n,48) the last state
      hidden,state = self.decoder2(Y_i,Y_fv,feature_map) 
      deltaY = self.fc_dy(hidden[-1]).view(state.shape[0],2,-1).permute(2,0,1)
      delta_Y_list[i] = deltaY
      score = torch.sum(self.fc_score(hidden),dim=0)
      scores[i] = score
    return Y_path, delta_Y_list, scores, H_miu,H_delta
  def compute_vel(self,path,current_location):
    '''
    path:tensor(40,n,2) 
    current_location:the current location for agents.Tensor with size(n,2)
    '''
    sequence = path.shape[0]
    vel = torch.zeros(path.shape)
    for j in range(sequence):
      if j==0:
        vel[j] = (path[j]-current_location)*self.hz
      else:
        vel[j] = (path[j]-path[j-1])*self.hz
    return vel
      # (40,n,2) and (n,2) to compute_vel

    pass
  def compute_dist_loss(self,Y_i,Y):
    '''
    Y_i:predict path:(K,40,n,2)
    Y  :ground truth: (n,2,40)
    '''
    loss_sum = 0.0
    Y_gt= Y.permute(2,0,1)
    for i in range(self.K):
      loss_sum += (Y_i[i]-Y_gt).norm()
    return loss_sum/self.K/Y.shape[1]

  def compute_kld(self,miu,sigma):
    '''
    miu:(n,48)
    sigma:(n,48)
    return :the loss of KLD:-0.5*(1+log(sigma*sigma)-sigma*sigma-miu*miu)
    '''
    sigma2 = torch.square(sigma)
    return torch.mean(-0.5*(1+torch.log(sigma2+1e-10)-sigma2-torch.square(miu)))

  def compute_cross_entropy(self,oldY,newY,Y):
    '''
    oldY:(K,40,n,2)
    newY:(K,40,n,2)
    Y:(n,2,40)
    '''
    Y_resize = Y.permute(2,0,1)
    loss = torch.zeros(1)
    for i in range(self.K):
      old = oldY[i]
      new = newY[i]
      #dist (40, n)
      old_d = torch.max(torch.abs(old-Y_resize),dim=2).values
      new_d = torch.max(torch.abs(new-Y_resize),dim=2).values
      # P, Q (40, n)
      P = func.softmax(old_d,dim=0)
      Q = func.softmax(new_d,dim=0)
      Hpq = torch.sum(-P*torch.log(Q))
      loss += Hpq
    return loss

  def compute_regression(self,Y_i,Y):
    '''
    Y_i:predict path:K list with (40,n,2)
    Y  :ground truth: (n,2,40)
    '''
    loss_sum = 0.0
    for i in range(self.K):
      loss_sum += (Y_i[i]-Y.permute(2,0,1)).norm()
    return loss_sum/self.K/Y.shape[1]

  def train(self, trajectory_data_x,trajectory_data_y,image_data):
    '''
    input:
    trajectory_data_x: a tensor with shape (n,2,20) n is the numbers of object
    trajectory_data_y: a tensor with shape (n,2,40) n is the numbers of object
    image_data: a tensor with shape (160,160,4)
    '''
    # print("begin train")
    # predict_path:(K,40, n, 2)
    # delta_y:(K,40, n, 2)
    # scores: (K,n, 1)
    # miu: (n, 48)
    # sigma: (n,48)
    predict_path,delta_y,scores,miu,sigma = self.forward(trajectory_data_x,trajectory_data_y,image_data)
    new_predict_path = predict_path+delta_y
    loss_distance = self.compute_dist_loss(predict_path,trajectory_data_y)
    loss_kld = self.compute_kld(miu,sigma)
    loss_ce = self.compute_cross_entropy(predict_path,new_predict_path,trajectory_data_y)
    loss_regression = self.compute_regression(new_predict_path,trajectory_data_y)
    loss = loss_distance+loss_kld+loss_ce+loss_regression
    return loss

  def build(self):
    self.rnn_encoder1 = nn.Sequential(nn.Conv1d(2,16,kernel_size=3,padding=1),
                                      nn.ReLU())
    self.encoder1_gru = nn.GRU(16,48,20)
    self.rnn_encoder2 = nn.Sequential(nn.Conv1d(2,16,kernel_size=1),
                                      nn.ReLU())
                                  
    self.encoder2_gru = nn.GRU(16,48,40)
    #must concat first: (48,48)->96
    self.fc1 = nn.Sequential(nn.Linear(96,48),nn.ReLU())
    self.fc2 = nn.Linear(48,48)
    self.fc3 = nn.Sequential(nn.Linear(48,48),HalfExp())
    self.fc4 = nn.Sequential(nn.Linear(48,48),nn.Softmax(dim=1))
    # multiplication
    self.sample_reconstruction = nn.GRU(48,48,40)
    self.fc5 = nn.Linear(48,2)
    
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
    self.decoder2 = SCF_GRU(self.K,96,48,40,radius_range=(0.5,4.0),social_pooling_size=(6,6),device=self.device) 
    # for output(48)->(2,40)
    self.fc_score = nn.Linear(48,1)
    self.fc_dy =nn.Sequential(nn.Linear(48,80),nn.ReLU())

if __name__ == '__main__':
  pass
