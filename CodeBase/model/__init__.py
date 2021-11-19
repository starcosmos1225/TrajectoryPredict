from .YNet import YNetTorch as YNet
from .YNet import YNetTorchNoGoal as YNetNoGoal
from .Transformer import IndividualTF
from .LSTM import SingleLSTM, DoubleLSTM
from .resTraj import resnetTraj
model_dict = {
    'ynet': YNet,
    'ynetnoGoal': YNetNoGoal,
    'transformer': IndividualTF,
    'singleLSTM': SingleLSTM, 
    'doubleLSTM': DoubleLSTM,
    'resnetTraj': resnetTraj,
}