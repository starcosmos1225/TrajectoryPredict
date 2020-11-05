#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable 
import torch.nn.functional as func 
import numpy as np
from DESIRE_model import CVAEModel, RefineModel
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
    parser.add_argument('--use_gpu',type=bool,default=False)
    cfg = parser.parse_args()
    train(cfg)

def train(cfg):
  train_data_x, train_data_y, train_img = load_data(cfg.file_dir,max_size=250)
  cvae_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  if cfg.use_gpu:
    refine_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  else:
    refine_device = torch.device("cpu")
  print("cvae device is {}".format(cvae_device))
  cvae_model = CVAEModel(sample_number=cfg.nums_sample,device=cvae_device, batch_size=cfg.batch_size)
  refine_model = RefineModel(sample_number=cfg.nums_sample,hz=cfg.frequent,device=refine_device, batch_size=cfg.batch_size)
  cvae_model.to(cvae_device)
  refine_model.to(refine_device)
  data_size = train_data_x.shape[0]
  #print(train_data_x.shape)
  #print(data_size)
  #order = np.arange(data_size)
  cvae_optimizer = torch.optim.Adam(cvae_model.parameters(),lr=cfg.learning_rate)
  refine_optimizer = torch.optim.Adam(refine_model.parameters(),lr=cfg.learning_rate)
  #data_size = train_data_x.shape[0]
  for epoch_i in range(cfg.epoch):
    print("the epoch is :{}".format(epoch_i))
    if epoch_i==0:
      cvae_model.zero_grad()
      refine_model.zero_grad()
    #np.random.shuffle(order)
    total_loss = torch.zeros(1)
    for index in range(0,data_size,cfg.batch_size):
      print("train index is :{}\r".format(index),end="")
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      start = time.time()
      # train_trajectory_x is [batch_size,n,2,20]
      train_trajectory_x = train_data_x[index:index+cfg.batch_size]
      current_location = train_trajectory_x.view(-1,train_trajectory_x.shape[2],train_trajectory_x.shape[3])[:, :, -1].detach()
      train_trajectory_x = train_trajectory_x.to(cvae_device)

      # train_trajectory_y is [batch_size,n,2,40] 
      train_trajectory_y = train_data_y[index:index+cfg.batch_size].to(cvae_device)
      
      # train_img_i is [batch_size,4,160,160]
      train_img_i = train_img[index:index+cfg.batch_size].to(refine_device)
      #hx: (batch_size*n,48)
      #y_path: (K,40,batch_size*n,2)
      hx,y_path,loss_cvae = cvae_model.train(train_trajectory_x, train_trajectory_y)
      #print(hx.shape)
      #print(y_path.shape)
      total_loss += loss_cvae.cpu()
      if not torch.cuda.is_available():
        hx = hx.to(refine_device).detach()
        y_path = y_path.to(refine_device).detach()
        loss_cvae.backward()
      
      #t=input('a')
      torch.nn.utils.clip_grad_norm_(cvae_model.parameters(),1.0)
      cvae_optimizer.step()
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      end = time.time()
      print("cvae time:{}".format(end-start))
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      start = time.time()
      loss_refine = refine_model.train(hx,current_location, y_path,train_img_i,train_data_y[index:index+cfg.batch_size])
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      end = time.time()
      print("refine train time:{}".format(end-start))
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      start = time.time()
      total_loss+= loss_refine.cpu()
      if not torch.cuda.is_available():
        loss_refine.backward()
      else:
        loss = loss_refine+loss_cvae
        loss.backward()
      torch.nn.utils.clip_grad_norm_(refine_model.parameters(),1.0)
      refine_optimizer.step()
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      end = time.time()
      print("the refine backward time:{}".format(end-start))
      # if torch.cuda.is_available():
      #   torch.cuda.synchronize()
      # start = time.time()
      # loss.backward()
      # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      # optimizer.step()
      # total_loss += loss.detach().item()
      # if torch.cuda.is_available():
      #   torch.cuda.synchronize()
      # end = time.time()
      # print("the post time:{}".format(end-start))
    if epoch_i %60==0:
      cvae_filename = cfg.save_dir+"cvae_{}.pth".format(epoch_i)
      refine_filename = cfg.save_dir+"refine_{}.pth".format(epoch_i)
      torch.save(cvae_model,cvae_filename)
      torch.save(refine_model,refine_filename)
    
    print("the total loss is :{}".format(total_loss[0]))
    gc.collect()


if __name__=='__main__':
  main()
  pass


