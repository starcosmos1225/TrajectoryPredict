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
	for _batch in batch:
		obs.append(_batch[0])
		gt.append(_batch[1])
		observedMap.append(_batch[2])
		gtFutureMap.append(_batch[3])
		gtWaypointMap.append(_batch[4])
		semanticMap.append(_batch[5])
	return torch.stack(obs),torch.stack(gt),\
		[torch.stack(observedMap),torch.stack(gtFutureMap), torch.stack(gtWaypointMap), torch.stack(semanticMap)]
