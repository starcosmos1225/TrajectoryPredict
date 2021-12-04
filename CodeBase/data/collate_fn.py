import torch



def traj_collate(batch):
    obs = []
    gt = []
    obsVelocity = []
    predVelocity = []
    gtPredVelocity = []
    # decInps = []
    # peds = []
    for _batch in batch:
		# print("batch device:{}".format(_batch[0].device))
        obs.append(_batch['obs'][:,:2])
        gt.append(_batch['pred'][:,:2])
        obsVelocity.append(_batch['obs'][1:,2:4].detach().clone())
        predVelocity.append(_batch['pred'][:-1,2:4].detach().clone())
        gtPredVelocity.append(_batch['pred'][:,2:4].detach().clone())

    
    return torch.stack(obs),torch.stack(gt), \
        [torch.stack(obsVelocity), torch.stack(predVelocity),torch.stack(gtPredVelocity)]

def PECtraj_collate(batch):
    obs = []
    gt = []

    for _batch in batch:
		# print("batch device:{}".format(_batch[0].device))
        obs.append(_batch['obs'][:,:2])
        gt.append(_batch['pred'][:,:2])
    
    return torch.stack(obs),torch.stack(gt), \
        [torch.stack(gt),torch.stack(obs)]



def scene_collate(batch):
    obs = []
    gt = []
    observedMap = []
    gtFutureMap = []
    gtWaypointMap = []
    semanticMap = []
    initTraj = []
    for _batch in batch:
        obs.append(_batch['obs'])
        gt.append(_batch['pred'])
        observedMap.append(_batch['observedMap'])
        gtFutureMap.append(_batch['gtFutureMap'])
        gtWaypointMap.append(_batch['gtWaypointMap'])
        if 'semanticMap' in _batch:
            semanticMap.append(_batch['semanticMap'])
        if 'initTraj' in _batch:
            initTraj.append(_batch['initTraj'])
    otherInfo = [torch.stack(observedMap), torch.stack(gtFutureMap), 
                torch.stack(gtWaypointMap)]
    if len(semanticMap)>0:
        otherInfo.append(torch.stack(semanticMap))
    if len(initTraj) >0:
        otherInfo.append(torch.stack(initTraj))
    otherInfo.append(torch.stack(gt))
    return torch.stack(obs),torch.stack(gt),otherInfo

def future_collate(batch):
    obs = []
    gt = []
    # observedMap = []
    gtFutureMap = []
    # gtWaypointMap = []
    semanticMap = []
    initTraj = []
    for _batch in batch:
        obs.append(_batch['obs'])
        gt.append(_batch['pred'])
        # observedMap.append(_batch['observedMap'])
        
        # gtWaypointMap.append(_batch['gtWaypointMap'])
        semanticMap.append(_batch['semanticMap'])
        gtFutureMap.append(_batch['gtFutureMap'])
        initTraj.append(_batch['initTraj'])
    otherInfo = [torch.stack(semanticMap),
                 torch.stack(gtFutureMap),
                 torch.stack(initTraj)]
    return torch.stack(obs),torch.stack(gt),otherInfo