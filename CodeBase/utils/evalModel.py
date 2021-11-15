from tqdm import tqdm
import torch
import time
def evalModel(params, dataloader, model, logger):
    model.eval()
    valADE = []
    valFDE = []
    device = params.device
    with torch.no_grad():
        # start = time.time()
        for obs, gt, otherInfo in tqdm(dataloader,desc='evaluate model'):
            # logger.info("eval loader time:{}".format(time.time()-start))
            # start = time.time()
			# Get scene image and apply semantic segmentation

            obs = obs.to(device)
            gt = gt.to(device)
            otherInp = []
            for info in otherInfo:
                otherInp.append(info.to(device))
            pred, otherOut = model(obs, otherInp, params)
            # logger.info("forward time:{}".format(time.time()-start))
            # start = time.time()
            waypointSamples = otherOut[0]
            gt_goal = gt[:, -1:]
            resize = params.dataset.resize
            # logger.info("gt shape:{}".format(gt.shape))
            # logger.info("pred shape:{}".format(pred.shape))
            # converts ETH/UCY pixel coordinates back into world-coordinates
            # if dataset_name == 'eth':
            #     waypoint_samples = image2world(waypoint_samples, scene, homo_mat, resize)
            #     pred_traj = image2world(pred_traj, scene, homo_mat, resize)
            #     gt_future = image2world(gt_future, scene, homo_mat, resize)

            valFDE.append(((((gt_goal.unsqueeze(0) - waypointSamples[:, :, -1:]) / resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
            valADE.append(((((gt.unsqueeze(0) - pred) / resize) ** 2).sum(dim=3) ** 0.5).mean(dim=2).min(dim=0)[0])
        valADE = torch.cat(valADE).mean()
        valFDE = torch.cat(valFDE).mean()
    return valADE.item(), valFDE.item()
