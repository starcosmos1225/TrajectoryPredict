import yaml
from easydict import EasyDict
import os
# os.environ['OMP_NUM_THREADS'] = '10'
import logging
from utils.evalModel import evalModel
from data import createDataLoader, createExtraInfo
import argparse
import warnings
from torch.serialization import SourceChangeWarning
import json
from model import model_dict
import torch

# os.environ['OPENBLAS_NUM_THREADS'] = 10
# os.environ['MKL_NUM_THREADS'] = 10

warnings.filterwarnings("ignore", category=SourceChangeWarning)


logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(funcName)s-%(lineno)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)



def main(params):
    testDataLoader = createDataLoader(params.dataset,type='test')
    # for idx, infos in enumerate(testDataLoader):
    #     print("\r {}/{}".format(idx,len(testDataLoader)))
    #     obs, gt,  otherInp = infos
    #     logger.info("obs:{}".format(obs.shape))
    #     logger.info("gt:{}".format(gt.shape))
    #     gtFutureMap,semanticMap,initTraj = otherInp
    #     # logger.info("observedMap:{}".format(observedMap.shape))
    #     logger.info("gtFutureMap:{}".format(gtFutureMap.shape))
    #     # logger.info("gtWaypointMap:{}".format(gtWaypointMap.shape))
    #     logger.info("semanticMap:{}".format(semanticMap.shape))
    #     logger.info("initTraj:{}".format(initTraj.shape))
    #     t=input()
    #     logger.info("observemap:{}".format(otherInp[0].shape))
    #     logger.info("gtFutre:{}".format(otherInp[1].shape))
    #     logger.info("waypoint:{}".format(otherInp[2].shape))
    #     logger.info("semantic:{}".format(otherInp[3].shape))
    # return
    model = model_dict[params.model.name](**params.model.kwargs)
    if params.model.pretrain !='' and os.path.exists(params.model.pretrain):
        model.load_state_dict(torch.load(params.model.pretrain))
    model.to(params.device)
    extraInfo = createExtraInfo(params,[testDataLoader])
    valADE, valFDE = evalModel(params, testDataLoader, model, extraInfo,logger,params.test.round)
    logger.info("ADE:{} FDE:{}".format(valADE, valFDE))




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