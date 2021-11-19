import yaml
from easydict import EasyDict
import os
import logging
from utils.evalModel import evalModel
from data import createDataLoader, createExtraInfo
import argparse
import warnings
from torch.serialization import SourceChangeWarning
import json
from model import model_dict

warnings.filterwarnings("ignore", category=SourceChangeWarning)


logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(funcName)s-%(lineno)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)



def main(params):
    testDataLoader = createDataLoader(params.dataset,type='test')
    # for idx, infos in enumerate(trainDataLoader):
    #     obs, gt,  otherInp = infos
    #     logger.info("obs:{}".format(obs.shape))
    #     logger.info("gt:{}".format(gt.shape))
    #     frames, seq_start, dataset, peds = otherInp
    #     logger.info("frames:{}".format(frames))
    #     logger.info("seq_start:{}".format(seq_start))
    #     logger.info("dataset:{}".format(dataset))
    #     logger.info("peds:{}".format(peds))
        # logger.info("observemap:{}".format(otherInp[0].shape))
        # logger.info("gtFutre:{}".format(otherInp[1].shape))
        # logger.info("waypoint:{}".format(otherInp[2].shape))
        # logger.info("semantic:{}".format(otherInp[3].shape))
    model = model_dict[params.model.name](**params.model.kwargs)
    if params.model.pretrain !='' and os.path.exists(params.model.pretrain):
        model.load(params.model.pretrain)
    # optimizer = createOptimizer(params,model)
    # if params.optim.name =='Noam':
        # optimizer.setWarmUpFactor(len(trainDataLoader))
    # lossFunction = createLossFunction(params)
    extraInfo = createExtraInfo(params,[testDataLoader])
    evalModel(params, testDataLoader, model, extraInfo,logger,params.test.round)




if __name__=='__main__':
    parser = argparse.ArgumentParser(description='train trajectory prediction task')
    parser.add_argument('--config', type=str, help='config file')
    args = parser.parse_args()
    with open(args.config) as file:
        params_dict = yaml.load(file, Loader=yaml.FullLoader)
    params = EasyDict(params_dict)
    experiment_name = params.experiment_name
    logger.info(json.dumps(params_dict,indent=2,ensure_ascii=False))
    main(params)