import torch
import torch.nn as nn
from utils.image_utils import get_patch, image2world


def trainModel(params, trainDataLoader,valDataLoader,model,optimizer, lossFunction,logger):
        model.to(params.device)
        model.train()
        maxEpochs = params.num_epochs
        for epoch in range(maxEpochs):
                train_loss = 0
                train_ADE = []
                train_FDE = []

                counter = 0
                # outer loop, for loop over each scene as scenes have different image size and to calculate segmentation only once
                for batch, infos in enumerate(trainDataLoader):
                        counter+=1
                        obs, gt,  observedMap, gtFutureMap, gtWaypointMap, semanticMap = infos
                        
                        featureInput = torch.cat([semanticMap, observedMap], dim=1)

                        features = model.pred_features(featureInput)

                        predGoalMap = model.pred_goal(features)
                        gtWaypointsMapsDownsampled = [nn.AvgPool2d(kernel_size=2**i, stride=2**i)(gtWaypointMap) for i in range(1, len(features))]
                        gtWaypointsMapsDownsampled = [gtWaypointMap] + gtWaypointsMapsDownsampled
                        
                        trajInput = [torch.cat([feature, goal], dim=1) for feature, goal in zip(features, gtWaypointsMapsDownsampled)]
                        predTrajMap = model.pred_traj(trajInput)
                        loss = lossFunction(predTrajMap,gtFutureMap,predGoalMap) # BCEWithLogitsLoss

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        with torch.no_grad():
                                train_loss += loss
                                # Evaluate using Softargmax, not a very exact evaluation but a lot faster than full prediction
                                predTraj = model.softargmax(predTrajMap)
                                predGoal = model.softargmax(predGoalMap[:, -1:])

                                # converts ETH/UCY pixel coordinates back into world-coordinates
                                # if dataset_name == 'eth':
                                #       pred_goal = image2world(pred_goal, scene, homo_mat, params)
                                #       pred_traj = image2world(pred_traj, scene, homo_mat, params)
                                #       gt_future = image2world(gt_future, scene, homo_mat, params)

                                train_ADE.append(((((gt - predTraj) / params.dataset.resize) ** 2).sum(dim=2) ** 0.5).mean(dim=1))
                                train_FDE.append(((((gt[:, -1:] - predGoal[:, -1:]) / params.dataset.resize) ** 2).sum(dim=2) ** 0.5).mean(dim=1))
                train_loss = train_loss / counter
                train_ADE = torch.cat(train_ADE).mean()
                train_FDE = torch.cat(train_FDE).mean()
                logger.info('Epoch {}/{} train ADE: {}  Val FDE: {} loss:{}'.format(epoch,maxEpochs,train_ADE.item(),train_FDE.item(),train_loss))
        return 
