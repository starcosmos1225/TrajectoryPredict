from .YNet import YNetTorch as YNet
from .YNet import YNetTorchNoGoal as YNetNoGoal
from .Transformer import IndividualTF
from .LSTM import SingleLSTM, DoubleLSTM,DoubleCVAELSTM, DoubleLSTMPure, DoubleLSTMGoal
from .resTraj import resnetTraj
from .PECNet import PECNet
model_dict = {
    'ynet': YNet,
    'ynetnoGoal': YNetNoGoal,
    'transformer': IndividualTF,
    'singleLSTM': SingleLSTM, 
    'doubleLSTM': DoubleLSTM,
    'doubleLSTMGoal': DoubleLSTMGoal,
    'doubleLSTMPure': DoubleLSTMPure,
    'doubleCVAELSTM': DoubleCVAELSTM,
    'resnetTraj': resnetTraj,
    'PECNet': PECNet
    
}