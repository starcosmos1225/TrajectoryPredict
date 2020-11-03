#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable 
import torch.nn.functional as func 
import numpy as np
from DESIRE_model import Model
from utils import load_data
import argparse
import gc 
import time

def main():
    '''
    Main function. Sets up all arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.004,
                        help='learning rate')
    parser.add_argument('--epoch', type=int, default=600,
                        help='nums of epoch')
    parser.add_argument('--nums_sample', type=int, default=10,
                        help='nums of sample')
    parser.add_argument('--frequent', type=int, default=10,
                        help='the frequent of frame')
    parser.add_argument('--save_dir', default='model/saved/')
    parser.add_argument('--file_dir', default='/home/hxy/Documents/TrajectoryPredict/code_for_DESIRE/data/train/')
    parser.add_argument('--batch_size',type=int,default=2)
    cfg = parser.parse_args()
    train(cfg)

def train(cfg):
  train_data_x, train_data_y, train_img = load_data(cfg.file_dir,max_size=50)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("device is {}".format(device))
  model = Model(sample_number=cfg.nums_sample,hz=cfg.frequent,device=device, batch_size=cfg.batch_size)
  optimizer = torch.optim.Adam(model.parameters(),lr=cfg.learning_rate)
  model.to(device)
  data_size = train_data_x.shape[0]
  #print(train_data_x.shape)
  #print(data_size)
  #order = np.arange(data_size)
  
  #data_size = train_data_x.shape[0]
  for epoch_i in range(cfg.epoch):
    print("the epoch is :{}".format(epoch_i))
    if epoch_i==0:
      model.zero_grad()
    #np.random.shuffle(order)
    total_loss = torch.zeros(1)
    for index in range(0,data_size,cfg.batch_size):
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      start = time.time()
      print("train index is :{}\r".format(index),end="")
      # train_trajectory_x is [batch_size,n,2,20]
      train_trajectory_x = train_data_x[index:index+cfg.batch_size].to(device)
      # train_trajectory_y is [batch_size,n,2,40] 
      train_trajectory_y = train_data_y[index:index+cfg.batch_size].to(device)
      
      # train_img_i is [batch_size,4,160,160]
      train_img_i = train_img[index:index+cfg.batch_size].to(device)
      loss = model.train(train_trajectory_x, train_trajectory_y, train_img_i)
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      start = time.time()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      total_loss += loss.detach().item()
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      end = time.time()
      print("the post time:{}".format(end-start))
    if epoch_i %60==0:
      filename = cfg.save_dir+"{}.pth".format(epoch_i)
      torch.save(model,filename)
    
    print("the total loss is :{}".format(total_loss[0]))
    gc.collect()


if __name__=='__main__':
  main()
  pass


