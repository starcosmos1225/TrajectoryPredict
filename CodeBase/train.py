import enum
import yaml
# from model import YNet,Transformer
from easydict import EasyDict
import os
import logging
# from utils.train import train
from data import createDataLoader
from optimizer import createOptimizer
from loss import createLossFunction
import argparse
import warnings
from torch.serialization import SourceChangeWarning
import json
from model import model_dict
warnings.filterwarnings("ignore", category=SourceChangeWarning)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def main(params):
    trainDataLoader, valDataLoader = createDataLoader(params)

    model = model_dict[params.model.name](**params.model.kwargs)
    if params.model.pretrain !='' and os.path.exists(params.model.pretrain):
        model.load(params.model.pretrain)
    optimizer = createOptimizer(params,model)
    lossFunction = createLossFunction(params)
    
    # trainTraj(params, model,optimizer, lossFunction)




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