import yaml
from easydict import EasyDict
import os
import logging
import argparse
import warnings
from torch.serialization import SourceChangeWarning
import json
from model import model_dict
import torch

warnings.filterwarnings("ignore", category=SourceChangeWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(funcName)s-%(lineno)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

def main(params,save_name):
    model = model_dict[params.model.name](**params.model.kwargs)
    if params.model.pretrain !='' and os.path.exists(params.model.pretrain):
        model.load_state_dict(torch.load(params.model.pretrain))
    torch.save(model,os.path.join("init_models",save_name))
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='train trajectory prediction task')
    parser.add_argument('--config', type=str, help='config file')
    parser.add_argument('--save_name', type=str,help='the full model name')
    args = parser.parse_args()
    save_name = args.save_name
    with open(args.config) as file:
        params_dict = yaml.load(file, Loader=yaml.FullLoader)
    params = EasyDict(params_dict)
    experiment_name = params.experiment_name
    logger.info(json.dumps(params_dict,indent=2,ensure_ascii=False))
    main(params,save_name)