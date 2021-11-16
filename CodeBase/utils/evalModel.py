from tqdm import tqdm
import torch
import time
from data.image_utils import createDistMat



def evalModel(params, dataloader, model, logger):
    model.eval()
    valADE = []
    valFDE = []
    device = params.device
    with torch.no_grad():
        # start = time.time()
        size = int(4200* params.dataset.resize)
        inputTemplate = createDistMat(size=size)
        inputTemplate = torch.from_numpy(inputTemplate).float().to(device)
        for obs, gt, otherInfo in tqdm(dataloader,desc='evaluate model'):
            # logger.info("eval loader time:{}".format(time.time()-start))
            # start = time.time()
            # logger.info("obs device:{}".format(obs.device))
			# Get scene image and apply semantic segmentation

            obs = obs.to(device)
            gt = gt.to(device)
            otherInp = []
            for info in otherInfo:
                otherInp.append(info.to(device))
            # logger.info("to device time:{}".format(time.time()-start))
            # start = time.time()
            otherInp.append(inputTemplate)
            pred, otherOut = model(obs, otherInp, params)
            # logger.info("forward time:{}".format(time.time()-start))
            # start = time.time()
            waypointSamples = otherOut[0]
            # logger.info("check shape")
            
            gt_goal = gt[:, -1:]
            # logger.info(gt.shape) # 4 12 2
            # logger.info(gt_goal.shape) # 4 1 2
            # logger.info(pred.shape) # 20 4 12 2
            resize = params.dataset.resize
            # logger.info("gt shape:{}".format(gt.shape))
            # logger.info("pred shape:{}".format(pred.shape))
            # converts ETH/UCY pixel coordinates back into world-coordinates
            # if dataset_name == 'eth':
            #     waypoint_samples = image2world(waypoint_samples, scene, homo_mat, resize)
            #     pred_traj = image2world(pred_traj, scene, homo_mat, resize)
            #     gt_future = image2world(gt_future, scene, homo_mat, resize)

            # valFDE.append(((((gt_goal.unsqueeze(0) - waypointSamples[:, :, -1:]) / resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
            valFDE.append(((((gt_goal.unsqueeze(0) - pred[:, :,-1:]) / resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
            valADE.append(((((gt.unsqueeze(0) - pred) / resize) ** 2).sum(dim=3) ** 0.5).mean(dim=2).min(dim=0)[0])
        valADE = torch.cat(valADE).mean()
        valFDE = torch.cat(valFDE).mean()
    return valADE.item(), valFDE.item()
