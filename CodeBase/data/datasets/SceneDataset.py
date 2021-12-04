from torch.autograd.grad_mode import F
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from ..image_utils import getPatch,createGaussianHeatmapTemplate, \
	 createDistMat, mapToRelativeVector, preprocessImageForSegmentation, \
		  pad, resize, nearestRelativeVector, croplocalImage
from ..preprocessing import augmentData, createImagesDict
import pandas as pd
import os
from easydict import EasyDict
from multiprocessing import Process
from multiprocessing import Manager
import time
import copy





class SceneDataset(Dataset):
	def __init__(self, params, mode='train'):
		""" Dataset that contains the trajectories of one scene as one element in the list. It doesn't contain the
		images to save memory.
		:params data (pd.DataFrame): Contains all trajectories
		:params resize (float): image resize factor, to also resize the trajectories to fit image scale
		:params total_len (int): total time steps, i.e. obs_len + pred_len
		"""
		super(SceneDataset,self).__init__()
		if mode =='train':
			dataPath = params.train_data_path
			imagePath = params.train_image_path
		elif mode =='val':
			dataPath = params.val_data_path
			imagePath = params.val_image_path
		elif mode=='test':
			dataPath = params.test_data_path
			imagePath = params.test_image_path
		else:
			raise ValueError("the mode:{} is invalid!".format(mode))
		if not dataPath.endswith('pkl'):
			raise ValueError('ImageTrajDataloader could only read the pkl data!')
		data = pd.read_pickle(dataPath)
		datasetName = params.name.lower()
		if datasetName == 'sdd':
			imageFileName = 'reference.jpg'
		elif datasetName == 'ind':
			imageFileName = 'reference.png'
        # elif datasetName == 'eth':
        #     imageFileName = 'oracle.png'
		else:
			raise ValueError('{} dataset is not supported'.format(datasetName))

        # ETH/UCY specific: Homography matrix is needed to convert pixel to world coordinates
		if datasetName == 'eth':
			homoMat = {}
			for scene in ['eth', 'hotel', 'students001', 'students003', 'uni_examples', 'zara1', 'zara2', 'zara3']:
				homoMat[scene] = torch.Tensor(np.loadtxt(f'data/eth_ucy/{scene}_H.txt'))
			segMask = True
		else:
			homoMat = None
			segMask = False

        # Load train images and augment train data and images
		if mode =='train':
			data, images = augmentData(data, image_path=imagePath, imageFile=imageFileName,
                                                                         segMask=segMask)
		else:
        	# Load val scene images
			images = createImagesDict(data, imagePath=imagePath, imageFile=imageFileName)
        # Preprocess images, in particular resize, pad and normalize as semantic segmentation backbone requires
		resize(images, factor=params.resize, segMask=segMask)
		pad(images, divisionFactor=params.division_factor)  # make sure that image shape is divisible by 32, for UNet segmentation
		preprocessImageForSegmentation(images, segMask=segMask)
        # Create template
		size = int(6200 * params.resize)

		inputTemplate = createDistMat(size=size)
		inputTemplate = torch.Tensor(inputTemplate)

		gtTemplate = createGaussianHeatmapTemplate(size=size, kernlen=params.kernlen, nsig=params.nsig, normalize=False)
		gtTemplate = torch.Tensor(gtTemplate)
		# print("dataset mid",flush=True)
		obsLength, predLength = params.obs_len, params.pred_len
		totalLength = obsLength + predLength
		# Load segmentation model

		if params.segmentation_model_fp!='' and os.path.exists(params.segmentation_model_fp):
			semanticModel = torch.load(params.segmentation_model_fp,map_location='cpu')
			if params.use_features_only:
				semanticModel.segmentation_head = nn.Identity()
            # semanticClasses = 16  # instead of classes use number of feature_dim
		else:
			semanticModel = None
		# Set Vairies
		# initTrajectoryModelFilePath
		if params.initTrajectoryModelFilePath!='' and os.path.exists(params.initTrajectoryModelFilePath):
			initTrajModel = torch.load(params.initTrajectoryModelFilePath,map_location='cpu')
		else:
			initTrajModel = None
		self.trajectories, self.scene_list = self.split_trajectories_by_scene(data, totalLength)
		self.trajectories = self.trajectories * params.resize
		self.obsLength = obsLength
		self.predLength = predLength
		self.inputTemplate = inputTemplate
		self.gtTemplate = gtTemplate
		self.waypoints = params.waypoints
		self.sceneImage = {}
		self.initTrajModel = initTrajModel
		self.num_traj = params.num_traj
		
		self.env_type = params.env_type
		if self.env_type =='local':
			self.crop_size = params.crop_size
		elif self.env_type=='rv':
			self.num_points = params.num_points
		if self.initTrajModel is not None:
			self.initTrajModel.eval()
		
		# print("dataset end",flush=True)
		for key in tqdm(images,desc='semantic image'):
			self.sceneImage[key] = images[key].unsqueeze(0)
			if semanticModel is not None:
				semanticModel.to(self.sceneImage[key].device)
				semanticModel.eval()
				self.sceneImage[key] = semanticModel.predict(self.sceneImage[key])
		
		if self.initTrajModel is not None:
				
				_,initmodelname = os.path.split(params.initTrajectoryModelFilePath)
				initmodelname,_ = os.path.splitext(initmodelname)
				_,dataname = os.path.split(dataPath)
				dataname,_ = os.path.splitext(dataname)
				modename = 'test' if mode!='train' else mode
				# relativeVectorName = os.path.join('temp','{}_{}_{}_{}.npy'.format('rv',initmodelname,dataname,modename))
				initTrajName = os.path.join('temp','{}_{}_{}_{}.npy'.format('traj',initmodelname,dataname,modename))
				if os.path.exists(initTrajName):
					self.initTrajList = np.load(initTrajName)
					# self.relativeVectorList = np.load(relativeVectorName)
				else:
					length = len(self.trajectories)
					self.initTrajList = [0 for _ in range(length)]
					if 'DoubleMLP' in params.initTrajectoryModelFilePath:
						traj_params = {
							'device': 'cpu',
							'dataset':{
								'pred_len': self.predLength,
								'num_traj':  self.num_traj
							}
						}
						mean = torch.tensor([0.0316,-0.3028])
						var = torch.tensor([5.888,3.5717])
						traj_params = EasyDict(traj_params)
						for idx in tqdm(range(length),desc='cal init traj'):
							obs = self.trajectories[idx,:self.obsLength,:2]
							vel = self.trajectories[idx,:self.obsLength,2:]
							obs = torch.from_numpy(obs).float()
							vel = torch.from_numpy(vel).float()
							with torch.no_grad():
								initTraj = self.initTrajModel(obs.unsqueeze(0),[vel.unsqueeze(0)], [mean, var], params=traj_params)
							self.initTrajList[idx] = initTraj.squeeze(1).detach().numpy()
					elif 'PEC' in params.initTrajectoryModelFilePath:
						traj_params = {
							'device': 'cpu',
							# 'num_points': params.num_points,
							'dataset':{
								'pred_len': self.predLength,
								'num_traj':  self.num_traj
							}
						}
						traj_params = EasyDict(traj_params)
						for idx in tqdm(range(length),desc='cal init traj'):
							obs = self.trajectories[idx,:self.obsLength,:2]
							obs = torch.from_numpy(obs).float()
							with torch.no_grad():
								initTraj = self.initTrajModel(obs.unsqueeze(0),params=traj_params)
							self.initTrajList[idx] = initTraj.squeeze(1).detach().numpy()
					self.initTrajList = np.array(self.initTrajList)
					np.save(initTrajName, self.initTrajList)
					
	def __len__(self):
		return len(self.trajectories)

	def computeRelativeVector(self,
						  params, 
						  index,
						  rank):
		if rank ==0:
			pbar = tqdm(index,desc='making init predict trajectories')
		else:
			pbar = index
		for idx in pbar:
			scene = self.scene_list[idx]
			_, _, H, W = self.sceneImage[scene].shape
			semanticMap = self.sceneImage[scene].view(-1,H, W)
			initTraj = self.initTrajList[idx]
			nearestVector = nearestRelativeVector(semanticMap.detach().numpy(), initTraj, params.num_points)
			self.relativeVectorList[idx] = nearestVector
			
	def __getitem__(self, idx):
		obsLength = self.obsLength
		predLength = self.predLength
		trajectory = self.trajectories[idx]
		# print("trajectory:{}".format(trajectory.shape))
		obs = trajectory[:obsLength,:2]
		gtFuture = trajectory[obsLength:,:2]
		
		scene = self.scene_list[idx]
		_, _, H, W = self.sceneImage[scene].shape
		observed = obs.reshape(-1, 2)
		
		observedMap = getPatch(self.inputTemplate, observed, H, W)
		# print("template:{} obs:{}".format(self.inputTemplate.shape,torch.stack(observedMap).shape))
		observedMap = torch.stack(observedMap).reshape([ obsLength, H, W])

		gtFutureMap = getPatch(self.gtTemplate, gtFuture.reshape(-1, 2), H, W)
		gtFutureMap = torch.stack(gtFutureMap).reshape([ self.predLength, H, W])
        
		gtWaypoints = gtFuture[self.waypoints]
		# print("way points:{}".format(gtWaypoints))
		gtWaypointMap = getPatch(self.inputTemplate, gtWaypoints.reshape(-1, 2), H, W)
		gtWaypointMap = torch.stack(gtWaypointMap).reshape([ gtWaypoints.shape[0], H, W])

		# Concatenate heatmap and semantic map
		semanticMap = self.sceneImage[scene].view(-1,H, W) 

		info = {
			'obs': torch.from_numpy(obs).float(),
			'pred': torch.from_numpy(gtFuture).float()
		}
		otherinfo = {
			'observedMap': observedMap,
			'gtFutureMap': gtFutureMap,
			'gtWaypointMap': gtWaypointMap,
			'semanticMap': semanticMap
		}
		if self.initTrajModel is not None:
			# print(self.relativeVectorList.shape)
			# print(self.relativeVectorList[idx].shape)
			# print(self.relativeVectorList)
			
			
			initTraj = self.initTrajList[idx]
			# if self.relativeVectorList is not None:
			# 	nearestVector = self.relativeVectorList[idx]
			# else:
			if self.env_type == 'rv':
				envInfo = nearestRelativeVector(semanticMap.detach().numpy(), initTraj, self.num_points)
				otherinfo['semanticMap'] = torch.from_numpy(envInfo).float()
				otherinfo['initTraj'] = torch.from_numpy(self.initTrajList[idx]).float()
			elif self.env_type == 'local':
				envInfo = croplocalImage(semanticMap.detach().numpy(), initTraj,self.crop_size)
				otherinfo['semanticMap'] = torch.from_numpy(envInfo).float()
				otherinfo['initTraj'] = torch.from_numpy(self.initTrajList[idx]).float()
			elif self.env_type == 'future':
				traj = initTraj.reshape(-1, 2)
				# print("traj")
				# print(traj.shape)
				trajMap = getPatch(self.inputTemplate, traj, H, W)
				# print("tramap")
				# print(trajMap[0].shape)
				trajMap = torch.stack(trajMap).reshape([-1,predLength, H, W])
				otherinfo['initTraj'] = trajMap
			
			
		res = info.copy()
		res.update(otherinfo)
		# print("device:{}".format(self.device))
		return res

	def getSamplerInfo(self):
		c = 0
		interval = []
		tempSceneId = None
		for sceneId in self.scene_list:
			if sceneId != tempSceneId and tempSceneId is not None:
				interval.append(c)
			c+=1
			tempSceneId = sceneId
		interval.append(len(self.trajectories))
		# print("interval:{}".format(interval))
		return {
			'length': self.__len__(),
			'interval': interval
		}

	def split_trajectories_by_scene(self, data, total_len):
		trajectoriesList = []
		metaList = []
		sceneList = []
		# print("type data:{}".format(type(data)))
		for meta_id, meta_df in tqdm(data.groupby('sceneId', as_index=False), desc='Prepare Dataset'):
			trajectoriesList.append(meta_df[['x', 'y']].to_numpy().astype('float32').reshape(-1, total_len, 2))
			metaList.append(meta_df)
			sceneList.append(meta_df.iloc()[0:1].sceneId.item())
		trajectory = []
		meta = []
		scene = []
		for traj,m,s in zip(trajectoriesList,metaList,sceneList):
			for i in range(traj.shape[0]):
				speed = np.concatenate((np.zeros((1,2)),traj[i][1:] - traj[i][ :-1]),0)
				traj_i =np.concatenate((traj[i],speed),1)
				trajectory.append(traj_i)
				meta.append(m)
				scene.append(s)
        
		return np.array(trajectory),  scene

# def scene_collate(batch):
# 	obs = []
# 	gt = []
# 	observedMap = []
# 	gtFutureMap = []
# 	gtWaypointMap = []
# 	semanticMap = []
# 	for _batch in batch:
# 		obs.append(_batch[0])
# 		gt.append(_batch[1])
# 		observedMap.append(_batch[2])
# 		gtFutureMap.append(_batch[3])
# 		gtWaypointMap.append(_batch[4])
# 		semanticMap.append(_batch[5])
# 	return torch.stack(obs),torch.stack(gt),\
# 		[torch.stack(observedMap),torch.stack(gtFutureMap), torch.stack(gtWaypointMap), torch.stack(semanticMap)]
