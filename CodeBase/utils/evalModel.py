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

                # otherInp.append(inputTemplate)
                pred, otherOut = model(obs, otherInp, extraInfo,params)

                # waypointSamples = otherOut[0]
                # logger.info("check shape")
                
                gt_goal = gt[:, -1:]

                resize = params.dataset.resize
                valFDE.append(((((gt_goal.unsqueeze(0) - pred[:, :,-1:]) / resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
                valADE.append(((((gt.unsqueeze(0) - pred) / resize) ** 2).sum(dim=3) ** 0.5).mean(dim=2).min(dim=0)[0])
    valADE = torch.cat(valADE).mean()
    valFDE = torch.cat(valFDE).mean()
    return valADE.item(), valFDE.item()
