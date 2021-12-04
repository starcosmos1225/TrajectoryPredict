import torch
import torch.nn as nn
from tqdm import tqdm
from time import time
from .evalModel import evalModel

def trainModel(params, trainDataLoader,valDataLoader,model,optimizer, lossFunction,extraInfo,logger):
        
        device = params.device
        
        bestTestADE = 999999999
        bestTestFDE = 999999999
        trainADERecord = []
        trainFDERecord = []
        valADERecord = []
        valFDERecord = []
        maxEpochs = params.optim.num_epochs
        for epoch in range(maxEpochs):
                # training
                # ''' 
                logger.info("{}/{} start training...".format(epoch,maxEpochs))
                counter = 0
                train_loss = 0
                train_ADE = []
                train_FDE = []
                model.train()
                # start = time()
                for infos in tqdm(trainDataLoader,desc="training data"):
                        # logger.info("dataloader time:{}".format(time()-start))
                        # start = time()
                        counter+=1
                        obs, gt,  otherInfo = infos
                        obs = obs.to(device)
                        gt = gt.to(device)
                        otherInp = []
                        for inp in otherInfo:
                                otherInp.append(inp.to(device))

                        pred, otherOut = model(obs,otherInp,extraInfo, params)
                        # print(pred.shape)
                        assert (len(pred.shape)==3 or len(pred.shape)==4) and len(gt.shape)==3, "Training model prediction trajectoies' shape must be (batch, squence, position) \
                         or (num_traj, batch_size, squence, position)"
                        # logger.info("predVel info:{}".format(otherOut[0].shape))
                        # logger.info(otherOut[0])
                        if len(pred.shape) ==3:
                                        pred = pred.unsqueeze(0)
                        loss = lossFunction(pred, gt.unsqueeze(0), otherInp, otherOut,extraInfo)
                        # logger.info("forward time:{}".format(time()-start))
                        # start = time()

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        # logger.info("backward time:{}".format(time()-start))
                        # start = time()
                        
                        with torch.no_grad():
                                train_loss += loss
                                # Evaluate using Softargmax, not a very exact evaluation but a lot faster than full prediction
                                # predGoal = otherOut[2]

                                # converts ETH/UCY pixel coordinates back into world-coordinates
                                # if dataset_name == 'eth':
                                #       pred_goal = image2world(pred_goal, scene, homo_mat, params)
                                #       pred_traj = image2world(pred_traj, scene, homo_mat, params)
                                #       gt_future = image2world(gt_future, scene, homo_mat, params)
                                # logger.info("pred shape:{} gt shape:{}".format(pred.shape,gt.shape))
                                # logger.info("gt:{}".format(gt[:, -1:]))
                                # logger.info("pred:{}".format(pred[:,-1:]))
                                # logger.info(((((gt[:, -1:] - pred[:,-1:]) / params.dataset.resize) ** 2).sum(dim=2) ** 0.5).mean(dim=0))
                                # t=input()
                                # print(gt)
                                # t=input()
                                # print(pred)
                                # t=input()
                                train_FDE.append(((((gt[:, -1:].unsqueeze(0) - pred[:, :,-1:,:]) / params.dataset.resize) ** 2).sum(dim=3) ** 0.5).min(dim=0)[0])
                                train_ADE.append(((((gt.unsqueeze(0) - pred) / params.dataset.resize) ** 2).sum(dim=3) ** 0.5).mean(dim=2).min(dim=0)[0])
                                # train_ADE.a/ppend(((((gt - pred) / params.dataset.resize) ** 2).sum(dim=2) ** 0.5).mean(dim=0))
                                # train_FDE.append(((((gt[:, -1:] - predGoal[:, -1:]) / params.dataset.resize) ** 2).sum(dim=2) ** 0.5).mean(dim=1))
                                # train_FDE.append(((((gt[:, -1:] - pred[:,-1:]) / params.dataset.resize) ** 2).sum(dim=2) ** 0.5).mean(dim=0))
                train_loss = train_loss / counter
                train_ADE = torch.cat(train_ADE).mean()
                train_FDE = torch.cat(train_FDE).mean()
                
                logger.info('Epoch {}/{} train ADE: {}  train FDE: {} loss:{}'.format(epoch,maxEpochs,train_ADE.item(),train_FDE.item(),train_loss))
                trainADERecord.append(train_ADE.item())
                trainFDERecord.append(train_FDE.item())
                # '''
                # begin eval
                if  epoch % params.test.eval_step ==0 or epoch == maxEpochs-1:
                        valADE, valFDE = evalModel(params,valDataLoader,model,extraInfo, logger)
                        valADERecord.append(valADE)
                        valFDERecord.append(valFDE)
                        logger.info('Epoch {}/{} val ADE: {}  val FDE: {}'.format(epoch,maxEpochs,valADE,valFDE))
                        if valADE<bestTestADE:
                                logger.info('Epoch {}/{} best val ADE: {}'.format(epoch,maxEpochs,valADE))
                                bestTestADE = valADE
                                bestTestFDE = valFDE
                                torch.save(model.state_dict(),'trained_models/{}.pth'.format(params.model.save_name))
                        logger.info('Epoch {}/{} best ADE: {}  best FDE: {}'.format(
                                epoch,maxEpochs,bestTestADE,bestTestFDE))
        logger.info('finish training and the  best val ADE: {} with FDE:{}'.format(bestTestADE, bestTestFDE))      
        return 
