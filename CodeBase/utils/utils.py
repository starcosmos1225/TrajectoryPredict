
import torch
from multiprocessing import Process
import numpy as np

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

def sampleIndex(mat, N):
    xy = np.random.choice(range(mat.size), size=N, p=mat.flatten())
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