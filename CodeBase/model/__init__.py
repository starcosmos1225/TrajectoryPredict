from .YNet import YNetTorch as YNet
from .Transformer import IndividualTF
from .LSTM import SingleLSTM, DoubleLSTM
from .resTraj import resnetTraj
model_dict = {
    'ynet': YNet,
    'transformer': IndividualTF,
    'singleLSTM': SingleLSTM, 
    'doubleLSTM': DoubleLSTM,
    'resnetTraj': resnetTraj
}