from torch.utils.data import Dataset, dataset
import os
import pandas as pd
from tqdm.std import tqdm
import numpy as np
import torch
import random
import scipy.spatial
import scipy.io
from ..preprocessing import augmentData


class TrajDataset:

    def __init__(self,params, mode='train'):
        if mode=='train':            
            datapath = params.train_data_path
            imagePath = params.train_image_path
        elif mode=='val':
            datapath = params.val_data_path
            imagePath = params.val_image_path
        elif mode=='test':
            datapath = params.test_data_path
            imagePath = params.test_image_path
        else:
            raise ValueError("the mode:{} is invalid!".format(mode))
        obsLength, predLength = params.obs_len, params.pred_len
        if os.path.isdir(datapath):
        
            datasetsList = os.listdir(datapath)
            data={}
            dataObs=[]
            dataPred=[]
            dataSeqStart=[]
            dataFrames=[]
            dataDt=[]
            dataPeds=[]

            for i_dt, dt in enumerate(tqdm(datasetsList,desc='preparing {}_data'.format(mode))):
                raw_data = pd.read_csv(os.path.join(datapath, dt), delimiter='\t',
                                                names=["frame", "ped", "x", "y"],usecols=[0,1,2,3],na_values="?")

                raw_data.sort_values(by=['frame','ped'], inplace=True)
                obs,pred,info=get_strided_data_clust(raw_data,obsLength, predLength,1)

                dtFrames=info['frames']
                dtSeqStart=info['seq_start']
                dtDataset=np.array([i_dt]).repeat(obs.shape[0])
                dtPeds=info['peds']

                dataObs.append(obs)
                dataPred.append(pred)
                dataSeqStart.append(dtSeqStart)
                dataFrames.append(dtFrames)
                dataDt.append(dtDataset)
                dataPeds.append(dtPeds)
        
            data['obs'] = np.concatenate(dataObs, 0)
            data['pred'] = np.concatenate(dataPred, 0)
            data['seq_start'] = np.concatenate(dataSeqStart, 0)
            data['frames'] = np.concatenate(dataFrames, 0)
            data['dataset'] = np.concatenate(dataDt, 0)
            data['peds'] = np.concatenate(dataPeds, 0)
            data['dataset_name'] = datasetsList

            self.mean= data['obs'].mean((0,1))
            self.std= data['obs'].std((0,1))
            self.data = data
            # print(data['dataset'].shape)
        else:
            if mode =='train':
                # dataPath = params.train_data_path
                imagePath = params.train_image_path
            elif mode =='val':
                # dataPath = params.val_data_path
                imagePath = params.val_image_path
            elif mode=='test':
                # dataPath = params.test_data_path
                imagePath = params.test_image_path
            if not datapath.endswith('pkl'):
                raise ValueError('ImageTrajDataloader could only read the pkl data!')
            data = pd.read_pickle(datapath)
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
                data, _ = augmentData(data, image_path=imagePath, imageFile=imageFileName,
                                                                                    segMask=segMask)

            self.data = {}
            
            obsLength, predLength = params.obs_len, params.pred_len
            totalLength = obsLength + predLength

            self.data['obs'], self.data['pred'],sceneList = self.split_trajectories_by_scene(data, totalLength,obsLength)
            self.obsLength = obsLength
            self.predLength = predLength
            self.data['dataset'] = sceneList
            # self.data['obs'] = self.trajectories[:,:obsLength,:]
            # self.data['pred'] = self.trajectories[:,obsLength:,:]
            
        self.data['obs'] = self.data['obs'] * params.resize
        self.data['pred'] = self.data['pred'] * params.resize

    def __len__(self):
        return self.data['obs'].shape[0]

    def getSamplerInfo(self):
        return {
            'length': self.__len__()
            }

            
    def __getitem__(self,index):
        return {'obs':torch.Tensor(self.data['obs'][index]),
                'pred':torch.Tensor(self.data['pred'][index]),
                # 'frames':self.data['frames'][index],
                # 'seq_start':self.data['seq_start'][index],
                'dataset': self.data['dataset'][index]
                # 'peds': self.data['peds'][index]
                }
    def split_trajectories_by_scene(self, data, total_len, obs_len):
        trajectoriesList = []
        metaList = []
        sceneList = []
        sceneMap = {}
        sceneId = data.sceneId.unique()
        for id, key in enumerate(sceneId):
            sceneMap[key] = id

        for meta_id, meta_df in tqdm(data.groupby('sceneId', as_index=False), desc='Prepare Dataset'):
            trajectoriesList.append(meta_df[['x', 'y']].to_numpy().astype('float32').reshape(-1, total_len, 2))
            metaList.append(meta_df)
            sceneList.append(sceneMap[meta_df.iloc()[0:1].sceneId.item()])
        
        obs = []
        pred = []
        # meta = []
        scene = []
        for traj,m,s in zip(trajectoriesList,metaList,sceneList):
            for i in range(traj.shape[0]):
                speed = np.concatenate((np.zeros((1,2)),traj[i][1:] - traj[i][ :-1]),0)
                traj_i =np.concatenate((traj[i],speed),1)
                obs_traj = traj_i[:obs_len]
                pred_traj = traj_i[obs_len:]
                obs.append(obs_traj)
                pred.append(pred_traj)
                # meta.append(m)
                scene.append(s)
        
        return np.array(obs),np.array(pred), np.array(scene)


def get_strided_data(dt, obs, pred, step):
    inp_te = []
    dtt = dt.astype(np.float32)
    raw_data = dtt

    ped = raw_data.ped.unique()
    frame=[]
    ped_ids=[]
    for p in ped:
        for i in range(1+(raw_data[raw_data.ped == p].shape[0] - obs - pred) // step):
            frame.append(dt[dt.ped == p].iloc[i * step:i * step + obs + pred, [0]].values.squeeze())
            inp_te.append(raw_data[raw_data.ped == p].iloc[i * step:i * step + obs + pred, 2:4].values)
            ped_ids.append(p)

    frames=np.stack(frame)
    inp_te_np = np.stack(inp_te)
    ped_ids=np.stack(ped_ids)

    inp_no_start = inp_te_np[:,1:,0:2] - inp_te_np[:, :-1, 0:2]
    inp_std = inp_no_start.std(axis=(0, 1))
    inp_mean = inp_no_start.mean(axis=(0, 1))
    inp_norm=inp_no_start

    return inp_norm[:,:obs-1],inp_norm[:,obs-1:],{'mean': inp_mean, 'std': inp_std, 'seq_start': inp_te_np[:, 0:1, :].copy(),'frames':frames,'peds':ped_ids}


def get_strided_data_2(dt, gt_size, horizon, step):
    inp_te = []
    dtt = dt.astype(np.float32)
    raw_data = dtt

    ped = raw_data.ped.unique()
    frame=[]
    ped_ids=[]
    for p in ped:
        for i in range(1+(raw_data[raw_data.ped == p].shape[0] - gt_size - horizon) // step):
            frame.append(dt[dt.ped == p].iloc[i * step:i * step + gt_size + horizon, [0]].values.squeeze())
            # print("%i,%i,%i" % (i * 4, i * 4 + gt_size, i * 4 + gt_size + horizon))
            inp_te.append(raw_data[raw_data.ped == p].iloc[i * step:i * step + gt_size + horizon, 2:4].values)
            ped_ids.append(p)

    frames=np.stack(frame)
    inp_te_np = np.stack(inp_te)
    ped_ids=np.stack(ped_ids)

    inp_relative_pos= inp_te_np-inp_te_np[:,:1,:]
    inp_speed = np.concatenate((np.zeros((inp_te_np.shape[0],1,2)),inp_te_np[:,1:,0:2] - inp_te_np[:, :-1, 0:2]),1)
    inp_accel = np.concatenate((np.zeros((inp_te_np.shape[0],1,2)),inp_speed[:,1:,0:2] - inp_speed[:, :-1, 0:2]),1)
    #inp_std = inp_no_start.std(axis=(0, 1))
    #inp_mean = inp_no_start.mean(axis=(0, 1))
    #inp_norm= inp_no_start
    #inp_norm = (inp_no_start - inp_mean) / inp_std

    #vis=inp_te_np[:,1:,2:4]/np.linalg.norm(inp_te_np[:,1:,2:4],2,axis=2)[:,:,np.newaxis]
    #inp_norm=np.concatenate((inp_norm,vis),2)
    inp_norm=np.concatenate((inp_te_np,inp_relative_pos,inp_speed,inp_accel),2)
    inp_mean=np.zeros(8)
    inp_std=np.ones(8)

    return inp_norm[:,:gt_size],inp_norm[:,gt_size:],{'mean': inp_mean, 'std': inp_std, 'seq_start': inp_te_np[:, 0:1, :].copy(),'frames':frames,'peds':ped_ids}

def get_strided_data_clust(dt, gt_size, horizon, step):
    inp_te = []
    dtt = dt.astype(np.float32)
    raw_data = dtt

    ped = raw_data.ped.unique()
    frame=[]
    ped_ids=[]
    datasetIndex = []
    # print(raw_data.keys())
    # t=input()
    createDataset = ('dataset' in raw_data.keys())
    # print("createData:{}".format(createDataset))
    # print(raw_data.keys)
    # t=input()
    for p in tqdm(ped, desc='preparing data'):
        for i in range(1+(raw_data[raw_data.ped == p].shape[0] - gt_size - horizon) // step):
            frame.append(dt[dt.ped == p].iloc[i * step:i * step + gt_size + horizon, [0]].values.squeeze())
            inp_te.append(raw_data[raw_data.ped == p].iloc[i * step:i * step + gt_size + horizon, 2:4].values)
            ped_ids.append(p)
            if createDataset:
                datasetIndex.append(raw_data[raw_data.ped==p].iloc[i * step, [4]].values)
    frames=np.stack(frame)
    inp_te_np = np.stack(inp_te)
    ped_ids=np.stack(ped_ids)

    inp_speed = np.concatenate((np.zeros((inp_te_np.shape[0],1,2)),inp_te_np[:,1:,0:2] - inp_te_np[:, :-1, 0:2]),1)
 
    inp_norm=np.concatenate((inp_te_np,inp_speed),2)

    inp_mean=np.zeros(4)
    inp_std=np.ones(4)
    
    out_dict = {
        'mean': inp_mean, 
        'std': inp_std, 
        'seq_start': inp_te_np[:, 0:1, :].copy(),
        'frames':frames,
        'peds':ped_ids
    }
    if createDataset:
        out_dict['dataset'] = datasetIndex

    return inp_norm[:,:gt_size], \
            inp_norm[:,gt_size:], \
            out_dict


def distance_metrics(gt,preds):
    errors = np.zeros(preds.shape[:-1])
    for i in range(errors.shape[0]):
        for j in range(errors.shape[1]):
            errors[i, j] = scipy.spatial.distance.euclidean(gt[i, j], preds[i, j])
    return errors.mean(),errors[:,-1].mean(),errors

def createDatasetFromSceneId(data):
    id = data.sceneId.unique()
    idMap = dict()
    for idx, sceneName in enumerate(id):
        idMap[sceneName] = idx
    for idx,sceneName in enumerate(tqdm(data.sceneId,desc='change scene to index')):
        data.loc[idx, 'sceneId'] = idMap[sceneName]
    # return sceneId