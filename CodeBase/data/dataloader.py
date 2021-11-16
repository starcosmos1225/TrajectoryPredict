from torch.autograd.grad_mode import F
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from .image_utils import getPatch
import torch

class SceneDataset(Dataset):
	def __init__(self, data, 
					   resize, 
					   obsLength,
					   predLength,
					   sceneImages,
					   inputTemplate = None,
					   gtTemplate = None,
					   waypoints = None,
					   semanticModel=None):
		""" Dataset that contains the trajectories of one scene as one element in the list. It doesn't contain the
		images to save memory.
		:params data (pd.DataFrame): Contains all trajectories
		:params resize (float): image resize factor, to also resize the trajectories to fit image scale
		:params total_len (int): total time steps, i.e. obs_len + pred_len
		"""
		totalLength = obsLength + predLength
		# self.split_trajectories(data, totalLength)
		self.trajectories, self.scene_list = self.split_trajectories_by_scene(data, totalLength)
		self.trajectories = self.trajectories * resize
		self.obsLength = obsLength
		self.predLength = predLength
		self.inputTemplate = inputTemplate
		self.gtTemplate = gtTemplate
		self.waypoints = waypoints
		self.sceneImage = {}
		# self.device = device
		# print("self device:{}".format(self.device))
		for key in tqdm(sceneImages,desc='semantic image'):
			self.sceneImage[key] = sceneImages[key].unsqueeze(0)
			if semanticModel is not None:
				semanticModel.eval()
				semanticModel.to(self.sceneImage[key].device)
				# print("input shape:{} key:{}".format(self.sceneImage[key].shape,key))
				self.sceneImage[key] = semanticModel.predict(self.sceneImage[key])
			# 	print("after seg:{}".format(self.sceneImage[key].shape))
		# print("finished",flush=True)
		# sceneImage = train_images[scene].to(device).unsqueeze(0)
		

	# def __len__(self):
	# 	return len(self.trajectories)

	def __getitem__(self, idx):
		obsLength = self.obsLength
		trajectory = self.trajectories[idx]
		# print("trajectory:{}".format(trajectory.shape))
		obs = trajectory[:obsLength,:]
		gtFuture = trajectory[obsLength:,:]
		
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
		# print("device:{}".format(self.device))
		return torch.from_numpy(obs),\
			torch.from_numpy(gtFuture),\
				 observedMap,\
				gtFutureMap,\
				gtWaypointMap, \
				semanticMap

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
		return interval

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
				trajectory.append(traj[i])
				meta.append(m)
				scene.append(s)
		return np.array(trajectory),  scene


# class scene_collate:
# 	def __init__(self,device):
# 		self.device=device

# 	def __call__(self,batch):
# 		obs = []
# 		gt = []
# 		observedMap = []
# 		gtFutureMap = []
# 		gtWaypointMap = []
# 		semanticMap = []
# 		for _batch in batch:
# 			print("batch device:{}".format(_batch[0].device))
# 			obs.append(_batch[0])
# 			gt.append(_batch[1])
# 			observedMap.append(_batch[2])
# 			gtFutureMap.append(_batch[3])
# 			gtWaypointMap.append(_batch[4])
# 			semanticMap.append(_batch[5])
# 		return torch.stack(obs).to(self.device),torch.stack(gt).to(self.device),\
# 			[torch.stack(observedMap).to(self.device),
# 			torch.stack(gtFutureMap).to(self.device), 
# 			torch.stack(gtWaypointMap).to(self.device), 
# 			torch.stack(semanticMap).to(self.device)]

def scene_collate(batch):
	obs = []
	gt = []
	observedMap = []
	gtFutureMap = []
	gtWaypointMap = []
	semanticMap = []
	for _batch in batch:
		# print("batch device:{}".format(_batch[0].device))
		obs.append(_batch[0])
		gt.append(_batch[1])
		observedMap.append(_batch[2])
		gtFutureMap.append(_batch[3])
		gtWaypointMap.append(_batch[4])
		semanticMap.append(_batch[5])
	return torch.stack(obs),torch.stack(gt),\
		[torch.stack(observedMap),torch.stack(gtFutureMap), torch.stack(gtWaypointMap), torch.stack(semanticMap)]
