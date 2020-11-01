#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable 
import torch.nn.functional as func 
import numpy as np
import os
trajector_dir = "trajectory/"
image_dir = "merged_data/"
def load_data(train_dir):
  '''
  return:
    trajectory_x:(size,n,2,20)
    trajectory_y:list[]->(size,n,2,40)
    data_img:list[]->(size,160,160,4)
  '''
  filedirs = os.listdir(train_dir)
  filedirs.sort()
  size = len(filedirs)
  trajectory_data_x = torch.zeros((size,10,2,20))
  trajectory_data_y = torch.zeros((size,10,2,40))
  data_img = torch.zeros((size,4,160,160))
  #max_size_n=0
  for file_i in range(len(filedirs)):
    # if (file_i>2):
    #   continue
    fn = filedirs[file_i]
    trajectory_path = os.path.join(train_dir, fn,trajector_dir)
    filenames = os.listdir(trajectory_path)
    trajectory_x = torch.zeros((10,2,20))
    trajectory_y = torch.zeros((10,2,40))
    #max_size_n = max(max_size_n,size_n)
    for index in range(len(filenames)):
      filename = filenames[index]
      filename = os.path.join(trajectory_path,filename)
      print(filename)
      fr = open(filename,'r')
      for i in range(20):
        l = fr.readline()
        l = l.strip().split()
        x = eval(l[0])
        y = eval(l[1])
        trajectory_x[index][0][i] = x
        trajectory_x[index][1][i] = y
      for i in range(40):
        l = fr.readline()
        l = l.strip().split()
        x = eval(l[0])
        y = eval(l[1])
        trajectory_y[index][0][i] = x
        trajectory_y[index][1][i] = y
    img_path = os.path.join(train_dir,fn,image_dir)
    filenames = os.listdir(img_path)
    filename = os.path.join(img_path ,filenames[0])
    img = torch.from_numpy(np.load(filename)).float().permute(2,0,1).unsqueeze(0)#.type('torch.DoubleTensor')
    data_img[file_i] = img
    trajectory_data_x[file_i] = trajectory_x
    trajectory_data_y[file_i] = trajectory_y
  return trajectory_data_x,trajectory_data_y,data_img

if __name__=='__main__':
  load_data()


