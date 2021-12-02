
import torch
from multiprocessing import Process
import numpy as np
import time

def sampling(probability_map, num_samples, rel_threshold=None, replacement=False):
	# new view that has shape=[batch*timestep, H*W]
	prob_map = probability_map.view(probability_map.size(0) * probability_map.size(1), -1)
	if rel_threshold is not None:
		thresh_values = prob_map.max(dim=1)[0].unsqueeze(1).expand(-1, prob_map.size(1))
		mask = prob_map < thresh_values * rel_threshold
		prob_map = prob_map * (~mask).int()
		prob_map = prob_map / prob_map.sum()

	# samples.shape=[batch*timestep, num_samples]
	samples = torch.multinomial(prob_map, num_samples=num_samples, replacement=replacement)
	# samples.shape=[batch, timestep, num_samples]

	# unravel sampled idx into coordinates of shape [batch, time, sample, 2]
	samples = samples.view(probability_map.size(0), probability_map.size(1), -1)
	idx = samples.unsqueeze(3)
	preds = idx.repeat(1, 1, 1, 2).float()
	preds[:, :, :, 0] = (preds[:, :, :, 0]) % probability_map.size(3)
	preds[:, :, :, 1] = torch.floor((preds[:, :, :, 1]) / probability_map.size(3))

	return preds

def torch_multivariate_gaussian_heatmap(coordinates, H, W, dist, sigma_factor, ratio, device, rot=False):
	"""
	Create Gaussian Kernel for CWS
	"""
	ax = torch.linspace(0, H, H, device=device) - coordinates[1]
	ay = torch.linspace(0, W, W, device=device) - coordinates[0]
	xx, yy = torch.meshgrid([ax, ay])
	meshgrid = torch.stack([yy, xx], dim=-1)
	radians = torch.atan2(dist[0], dist[1])

	c, s = torch.cos(radians), torch.sin(radians)
	R = torch.Tensor([[c, s], [-s, c]]).to(device)
	if rot:
		R = torch.matmul(torch.Tensor([[0, -1], [1, 0]]).to(device), R)
	dist_norm = dist.square().sum(-1).sqrt() + 5  # some small padding to avoid division by zero

	conv = torch.Tensor([[dist_norm / sigma_factor / ratio, 0], [0, dist_norm / sigma_factor]]).to(device)
	conv = torch.square(conv)
	T = torch.matmul(R, conv)
	T = torch.matmul(T, R.T)

	kernel = (torch.matmul(meshgrid, torch.inverse(T)) * meshgrid).sum(-1)
	kernel = torch.exp(-0.5 * kernel)
	return kernel / kernel.sum()


def nearestKindex(positions, origin, K):
	'''
	positions: n, 2
	origin: numtraj, squence,2
	K: nums of nearest points int
	return: numtraj, squence, K, 2
	'''
	# origin -> numtraj, squence , n,2
	# dist = (positions-origin)**2.sum(-1)**0.5
	# start = time.time()
	n = positions.shape[0]
	numTraj, numSquence,_ = origin.shape
	
	o = np.expand_dims(origin,axis=2)
	o = o.repeat(n,axis=2)
	# o = np.expand_dims(o,axis=2)
	# o = o.repeat(class_num,axis=2)
	# print(o.shape)
	# t=input()
	rv = positions - o
	# print("expand time:{}".format(time.time()-start))
	# start = time.time()
	absRv = np.abs(rv)
	dist = absRv[:,:,:,0] + absRv[:,:,:,1]
	# print(dist.shape)
	# dist = np.sum(np.abs(rv),axis=-1)
	# dist = ((rv**2).sum(-1))
	# print("dist:{}".format(dist.shape))
	# print("dist time:{}".format(time.time()-start))
	# start = time.time()
	index = np.argpartition(dist, K,axis=-1)
	# print(index.shape)
	# print("apart time:{}".format(time.time()-start))
	# start = time.time()
	index = index[:,:,:K]
	index = np.expand_dims(index,axis=-1)
	index = index.repeat(2,axis=-1)
	# print("indexshape{}".format(index.shape))
	rv = rv[np.arange(numTraj)[:,None,None,None],
			np.arange(numSquence)[None,:,None,None],
			index,
			np.arange(2)[None,None,None,:]]
	# print("rv time:{}".format(time.time()-start))

	# print("rvshape:{}".format(rv.shape))
	# t=input()
	# print("indexshape{}".format(index.shape))
	# rv = rv.gather(dim=2,index=index)
	# print("rvshapelater:{}".format(rv.shape))
	# t=input()
	return rv


def sampleIndex(mat, N):
    xy = np.random.choice(range(mat.size), size=N,replace=True, p=mat.flatten())
    return np.unravel_index(xy, mat.shape)

def choiceIndex(mat, index, outMat,N):
	# mat, index, outMat = inf
	pred = index
	for y in range(pred):
		m = mat[y]
		outMat[:,y,0], outMat[:,y,1] = sampleIndex(m, N)


def samplingTrajFromHeatMap(mat,N):
	'''
	mat: (batch, predlength, H,W)
	N: sample numbers
	return: (batch, predlength,2)
	'''
	# print(mat.shape)
	b, pred,H,W = mat.shape
	res = np.zeros((N,b,pred,2))
	# infos = []
	threads = []
	for i in range(b):
		# choiceIndex(mat[i],pred, res[:,i,:,:],N)
		p = Process(target=choiceIndex, args=(mat[i], pred, res[:,i,:,:],N))
		p.start()
		threads.append(p)
	for p in threads:
		p.join()
		
	# for i in range(b):
	# 	infos.append((mat[i], pred, res[i]))
	# with Pool(8) as p:
	# 	p.map(choiceIndex, infos)
	# print(res)
	return res