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
            # start = time.time()
            for obs, gt, otherInfo in tqdm(dataloader,desc='evaluate model'):
                # logger.info("loader time:{}".format(time.time()-start))
                # start = time.time()
                # Get scene image and apply semantic segmentation

                obs = obs.to(device)
                gt = gt.to(device)
                otherInp = []
                for info in otherInfo:
                    otherInp.append(info.to(device))

                pred= model(obs, otherInp, extraInfo,params)
                # logger.info("forward time:{}".format(time.time()-start))
                # start = time.time()
                assert len(gt.shape) ==3, "when evaluating, ground truth shape must be:(batch,squence,position)"

                gt_goal = gt[:, -1:]

                if len(pred.shape) ==3:
                    pred = pred.unsqueeze(0)
                
                resize = params.dataset.resize

                valFDE.append(((((gt_goal.unsqueeze(0) - pred[:, :,-1:,:]) / resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
                valADE.append(((((gt.unsqueeze(0) - pred) / resize) ** 2).sum(dim=3) ** 0.5).mean(dim=2).min(dim=0)[0])
                # logger.info("eval time:{}".format(time.time()-start))
                # start = time.time()
    valADE = torch.cat(valADE).mean()
    valFDE = torch.cat(valFDE).mean()
    return valADE.item(), valFDE.item()
