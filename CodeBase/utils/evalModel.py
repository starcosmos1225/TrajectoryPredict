from tqdm import tqdm
import torch
import time




def evalModel(params, dataloader, model,extraInfo, logger, rounds=1):
    model.eval()
    valADE = []
    valFDE = []
    device = params.device
    for _ in tqdm(range(rounds),desc='Round'):
        with torch.no_grad():
            for obs, gt, otherInfo in tqdm(dataloader,desc='evaluate model'):

                # Get scene image and apply semantic segmentation

                obs = obs.to(device)
                gt = gt.to(device)
                otherInp = []
                for info in otherInfo:
                    otherInp.append(info.to(device))

                pred= model(obs, otherInp, extraInfo,params)
                assert len(gt.shape) ==3, "when evaluating, ground truth shape must be:(batch,squence,position)"
                # waypointSamples = otherOut[0]
                # print(waypointSamples.shape)
                # print("way point:{}".format(waypointSamples[:,  -1:]))
                gt_goal = gt[:, -1:]
                # print("gt goal:{}".format(gt_goal))
                # print("pred.shape:{}".format(len(pred.shape)))
                if len(pred.shape) ==3:
                    pred = pred.unsqueeze(0)
                
                resize = params.dataset.resize
                # print("pred goal:{}".format(pred[:,:,-1:,:]))
                # print("goal:{}".format(gt_goal))
                # print("min:{}".format(((((gt_goal.unsqueeze(0) - pred[:, :,-1:,:]) / resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)))
                # t=input()
                # print(((((gt.unsqueeze(0) - pred) / resize) ** 2).sum(dim=3) ** 0.5).mean(dim=2).min(dim=0)[0].shape)
                # t=input()
                valFDE.append(((((gt_goal.unsqueeze(0) - pred[:, :,-1:,:]) / resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
                valADE.append(((((gt.unsqueeze(0) - pred) / resize) ** 2).sum(dim=3) ** 0.5).mean(dim=2).min(dim=0)[0])
    valADE = torch.cat(valADE).mean()
    valFDE = torch.cat(valFDE).mean()
    return valADE.item(), valFDE.item()
