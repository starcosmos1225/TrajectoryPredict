# from tqdm import cli
import numpy as np
import torch
import cv2
import torch.nn.functional as F
import time
# import os
from utils.utils import sampleIndex, nearestKindex
from torch import nn


def softmax2d(x):
	max = np.max(x, axis=1,keepdims=True)
	e_x = np.exp(x - max)
	sum = np.sum(e_x, axis=1, keepdims=True)
	f_x = e_x / sum
	return f_x

def gkern(kernlen=31, nsig=4):
	"""	creates gaussian kernel with side length l and a sigma of sig """
	ax = np.linspace(-(kernlen - 1) / 2., (kernlen - 1) / 2., kernlen)
	xx, yy = np.meshgrid(ax, ax)
	kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(nsig))
	return kernel / np.sum(kernel)


def createGaussianHeatmapTemplate(size, kernlen=81, nsig=4, normalize=True):
	""" Create a big gaussian heatmap template to later get patches out """
	template = np.zeros([size, size])
	kernel = gkern(kernlen=kernlen, nsig=nsig)
	m = kernel.shape[0]
	x_low = template.shape[1] // 2 - int(np.floor(m / 2))
	x_up = template.shape[1] // 2 + int(np.ceil(m / 2))
	y_low = template.shape[0] // 2 - int(np.floor(m / 2))
	y_up = template.shape[0] // 2 + int(np.ceil(m / 2))
	template[y_low:y_up, x_low:x_up] = kernel
	if normalize:
		template = template / template.max()
	return template


def createDistMat(size, normalize=True):
	""" Create a big distance matrix template to later get patches out """
	middle = size // 2
	distMat = np.linalg.norm(np.indices([size, size]) - np.array([middle, middle])[:,None,None], axis=0)
	if normalize:
		distMat = distMat / distMat.max() * 2
	return distMat


def getPatch(template, traj, H, W):
	# start = time.time()
	x = np.round(traj[:,0]).astype('int')
	y = np.round(traj[:,1]).astype('int')
	# print("get patch")
	# print(x.shape)
	# print(y.shape)
	x_low = template.shape[1] // 2 - x
	x_up = template.shape[1] // 2 + W - x
	y_low = template.shape[0] // 2 - y
	y_up = template.shape[0] // 2 + H - y
	# print(y_up>H)
	# print(x_up>W)

	patch = [template[y_l:y_u, x_l:x_u] for x_l, x_u, y_l, y_u in zip(x_low, x_up, y_low, y_up)]
	# print("get path inner:{}".format(time.time()-start))
	# print(patch[0].shape)
	return patch


def preprocessImageForSegmentation(images, encoder='resnet101', encoderWeights='imagenet', segMask=False, classes=6):
	""" Preprocess image for pretrained semantic segmentation, input is dictionary containing images
	In case input is segmentation map, then it will create one-hot-encoding from discrete values"""
	import segmentation_models_pytorch as smp

	preprocessingFn = smp.encoders.get_preprocessing_fn(encoder, encoderWeights)
	for key, im in images.items():
		if segMask:
			im = [(im == v) for v in range(classes)]
			im = np.stack(im, axis=-1)  # .astype('int16')
		else:
			# print(key)
			im = preprocessingFn(im)
		im = im.transpose(2, 0, 1).astype('float32')
		im = torch.Tensor(im)
		images[key] = im


def resize(images, factor, segMask=False):
	for key, image in images.items():
		if segMask:
			images[key] = cv2.resize(image, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
		else:
			images[key] = cv2.resize(image, (0,0), fx=factor, fy=factor, interpolation=cv2.INTER_AREA)


def pad(images, divisionFactor=32):
	""" Pad image so that it can be divided by division_factor, as many architectures such as UNet needs a specific size
	at it's bottlenet layer"""
	for key, im in images.items():
		if im.ndim == 3:
			H, W, C = im.shape
		else:
			H, W = im.shape
		H_new = int(np.ceil(H / divisionFactor) * divisionFactor)
		W_new = int(np.ceil(W / divisionFactor) * divisionFactor)
		im = cv2.copyMakeBorder(im, 0, H_new - H, 0, W_new - W, cv2.BORDER_CONSTANT)
		images[key] = im



def image2world(image_coords, scene, homo_mat, resize):
	traj_image2world = image_coords.clone()
	if traj_image2world.dim() == 4:
		traj_image2world = traj_image2world.reshape(-1, image_coords.shape[2], 2)
	if scene in ['eth', 'hotel']:
		traj_image2world[:, :, [0, 1]] = traj_image2world[:, :, [1, 0]]
	traj_image2world = traj_image2world / resize
	traj_image2world = F.pad(input=traj_image2world, pad=(0, 1, 0, 0), mode='constant', value=1)
	traj_image2world = traj_image2world.reshape(-1, 3)
	traj_image2world = torch.matmul(homo_mat[scene], traj_image2world.T).T
	traj_image2world = traj_image2world / traj_image2world[:, 2:]
	traj_image2world = traj_image2world[:, :2]
	traj_image2world = traj_image2world.view_as(image_coords)
	return traj_image2world

def to_shape(a, shape):
	'''
	a: numpy B,H,W
	shape: tuple new_H, new_W
	'''
	y_, x_ = shape
	_, y, x = a.shape
	y_pad = (y_-y)
	x_pad = (x_-x)
	return np.pad(a,((0,0),(y_pad//2, y_pad//2 + y_pad%2),
					(x_pad//2, x_pad//2 + x_pad%2)),
				mode = 'constant')

def croplocalImage(image, traj, size):
	'''
	image: num_class, H, W
	traj: num_traj, squence, 2
	size: int
	return: num_traj, squence, num_class, size,size
	'''
	num_class,H,W = image.shape
	num_traj, squence,_ = traj.shape
	cropImage = np.zeros((num_traj, squence,num_class,size,size))
	R = size//2
	for i in range(num_traj):
		for j in range(squence):
			x,y = traj[i,j,0], traj[i,j,1]
			l,t= int(x-R),int(y-R)
			l = max(0,l)
			t = max(0,t)
			r = l+size
			b = t+size
			r = min(H,r)
			b = min(W,b)
			im = image[:,t:b,l:r]
			if im.shape[1]!=size or im.shape[2]!=size:
				try:
					im = to_shape(im,(size,size))
				except ValueError as e:
					print(e)
					print(im.shape)
			cropImage[i,j] = im
	return cropImage




def nearestRelativeVector(image, traj, num_samples=512):
	'''
	for i: num_traj
		for j: squence
			for k: num_class:
				mat[i,j,k] = nearest(image[k], traj[i][j],num_samples)
	image: num_class, H, W
	traj: num_traj, squence, 2
	return: num_traj, squence, num_class, 512,2
	'''
	# start = time.time()
	C,H,W = image.shape
	numTraj, numSquence,_ = traj.shape
	zeros = np.zeros_like(image)
	# print(image.max())
	# t=input()
	clippedImage = np.where(image<0.95,zeros,image)
	rv = []

	for i in range(C):
		nonzero = np.transpose(clippedImage[i].nonzero())
		if nonzero.shape[0]>20000:
			idx = np.random.randint(nonzero.shape[0],size=20000)
			nonzero = nonzero[idx,:]
		
		if nonzero.shape[0] == 0:
			nonzero = np.ones((num_samples+1,2))*4098
		else:
			if nonzero.shape[0] < num_samples:
				nonzero = np.repeat(nonzero,num_samples//nonzero.shape[0]+1,axis=0)
		nearestVector = nearestKindex(nonzero, traj, K=num_samples)
		rv.append(nearestVector)

	rv = np.array(rv).transpose(1,2,0,3,4)
	return rv

def mapToRelativeVector(image, traj, num_samples=512):
	'''
	for i:num_traj:
		for j:squence:
			for k:num_class:
				mat[i,j,k] = samples(image[k],traj[i][j],num_samples)

	image: num_class, H, W
	traj: num_traj, squence, 2
	return: num_traj, squence,num_class,  512, 2
	  
	'''

	# image to sample: ->1,1,num_class, 512, 2
	C,H,W = image.shape
	zeros = torch.zeros_like(image)
	clippedImage = torch.where(image<0.5,zeros,image).view(C,-1).numpy()
	clippedImage = softmax2d(clippedImage)
	samples = np.zeros((1,1,C, num_samples, 2))
	for i in range(C):
		m = clippedImage[i].reshape(H,W)
		samples[0,0,i,:,0], samples[0,0,i,:,1] = sampleIndex(m,512)
	samples = torch.from_numpy(samples).float()
	# traj->repeat to -> num_traj,squence, num_class, 512, 2
	tileTraj = traj.unsqueeze(2).unsqueeze(2).repeat(1,1,image.shape[0], num_samples,1)
	# ralativeVector = sample - traj
	relativeVector = samples - tileTraj
	return relativeVector

