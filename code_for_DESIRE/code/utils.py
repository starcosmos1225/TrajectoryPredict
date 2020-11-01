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
    trajectory_x:list[]->(n,2,20)
    trajectory_y:list[]->(n,2,40)
    data_img:list[]->(160,160,4)
  '''
  filedirs = os.listdir(train_dir)
  filedirs.sort()
  trajectory_data_x = []
  trajectory_data_y = []
  data_img = []
  for file_i in range(len(filedirs)):
    # if (file_i>2):
    #   continue
    fn = filedirs[file_i]
    trajectory_path = os.path.join(train_dir, fn,trajector_dir)
    filenames = os.listdir(trajectory_path)
    size_n = len(filenames)
    trajectory_x = torch.zeros((size_n,2,20))
    trajectory_y = torch.zeros((size_n,2,40))
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
    #print(type(img))
    #print(img.type())
    #print(img)
    #t=input()
    #print(img.shape)
    data_img.append(img)
    #t=input()
    trajectory_data_x.append(trajectory_x)
    trajectory_data_y.append(trajectory_y)
  return trajectory_data_x,trajectory_data_y,data_img

if __name__=='__main__':
  load_data()


