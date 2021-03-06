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
    parser.add_argument('--save_filename', default='model/saved/loss_record.txt')
    parser.add_argument('--load_cvae_dir',default='None')
    parser.add_argument('--load_refine_dir',default='None')
    parser.add_argument('--file_dir', default='/home/hxy/Documents/TrajectoryPredict/code_for_DESIRE/data/train/')
    parser.add_argument('--batch_size',type=int,default=2)
    parser.add_argument('--use_gpu',action='store_true')
    cfg = parser.parse_args()
    train(cfg)

def train(cfg):
  #torch.autograd.set_detect_anomaly(True)
  train_data_x, train_data_y, train_img = load_data(cfg.file_dir,max_size=250)
  if cfg.use_gpu:
    print("to gpu")
    cvae_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    refine_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  else:
    print("to cpu")
    cvae_device = torch.device("cpu")
    refine_device = torch.device("cpu")
  if cfg.load_cvae_dir=='None':
    cvae_model = CVAEModel(sample_number=cfg.nums_sample,device=cvae_device, batch_size=cfg.batch_size)
  else:
    print("load cvae mode from:{}".format(cfg.load_cvae_dir))
    cvae_model = torch.load(cfg.load_cvae_dir)
  if cfg.load_refine_dir =='None':
    refine_model = RefineModel(sample_number=cfg.nums_sample,hz=cfg.frequent,device=refine_device, batch_size=cfg.batch_size)
  else:
    print("load refine mode from:{}".format(cfg.load_refine_dir))
    refine_model = torch.load(cfg.load_refine_dir)
  cvae_model.to(cvae_device)
  cvae_model.init(sample_number=cfg.nums_sample,device=cvae_device, batch_size=cfg.batch_size)
  refine_model.to(refine_device)
  refine_model.init(sample_number=cfg.nums_sample,hz=cfg.frequent,device=refine_device, batch_size=cfg.batch_size)
  # for p in cvae_model.parameters():
  #   p.data.fill_(0.001)
  # for p in refine_model.parameters():
  #   p.data.fill_(0.001)
  data_size = train_data_x.shape[0]
  #print(train_data_x.shape)
  #print(data_size)
  #order = np.arange(data_size)
  cvae_optimizer = torch.optim.Adam(cvae_model.parameters(),lr=cfg.learning_rate)
  refine_optimizer = torch.optim.Adam(refine_model.parameters(),lr=cfg.learning_rate)
  #data_size = train_data_x.shape[0]
  filename = cfg.save_filename
  fr = open(filename,"w")
  for epoch_i in range(cfg.epoch):
    print("{}:".format(epoch_i),end=" ")
    if epoch_i==0:
      cvae_model.zero_grad()
      refine_model.zero_grad()
    #np.random.shuffle(order)
    total_loss = 0.0
    for index in range(0,data_size,cfg.batch_size):
      #print("train index is :{}\r".format(index),end="")
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
      #print("cvae loss:{}".format(loss_cvae.cpu().item()))
      #print(hx.shape)
      #print(y_path.shape)
      total_loss += loss_cvae.cpu().item()
      #if not torch.cuda.is_available():
      loss_cvae.backward()
      #t=input('a')
      torch.nn.utils.clip_grad_norm_(cvae_model.parameters(),1.0)
      cvae_optimizer.step()
      hx = hx.to(refine_device).detach()
      y_path = y_path.to(refine_device).detach()
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      end = time.time()
      #print("cvae time:{}".format(end-start))
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      start = time.time()
      #y_test = torch.randn((10,40,50,2))
      #hx_test = torch.randn((hx.shape))
      
      loss_refine = refine_model.train(hx.detach(),current_location, y_path.detach(),train_img_i,train_data_y[index:index+cfg.batch_size])
      #print("refine loss:{}".format(loss_refine.cpu().item()))
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      end = time.time()
      #print("refine train time:{}".format(end-start))
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      start = time.time()
      #print("refine loss:{}".format(loss_refine.cpu().item()))
      total_loss+= loss_refine.cpu().item()
      #if not torch.cuda.is_available():
        #loss_refine.backward()
      #else:
      #loss = loss_refine+loss_cvae
      loss_refine.backward()
      torch.nn.utils.clip_grad_norm_(refine_model.parameters(),1.0)
      refine_optimizer.step()
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      end = time.time()
      #print("the refine backward time:{}".format(end-start))
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
    if epoch_i %10==0:
      cvae_filename = cfg.save_dir+"cvae_{}.pth".format(epoch_i)
      refine_filename = cfg.save_dir+"refine_{}.pth".format(epoch_i)
      torch.save(cvae_model,cvae_filename)
      torch.save(refine_model,refine_filename)
    
    print("the total loss is :{}".format(total_loss))
    fr.write("{}\n".format(total_loss))
    # gc.collect()
  torch.save(cvae_model,cvae_filename)
  torch.save(refine_model,refine_filename)
  fr.close()


if __name__=='__main__':
  main()
  pass


