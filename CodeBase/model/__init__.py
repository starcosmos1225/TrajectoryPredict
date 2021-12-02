from .YNet import YNetTorch as YNet
from .YNet import YNetTorchNoGoal as YNetNoGoal
from .Transformer import IndividualTF
from .LSTM import SingleLSTM, DoubleLSTM,DoubleCVAELSTM, DoubleLSTMPure,  \
    CVAEDoubleLSTM, CVAEMLPLSTM
from .resTraj import resnetTraj
from .PECNet import PECNet
from .NonSquence import CVAEDoubleMLP
from .RVNet import RVNet, RVNetResdual, RVNetTransformer,RVNetConv1d
from .localNet import LocalNet
model_dict = {
    'ynet': YNet,
    'ynetnoGoal': YNetNoGoal,
    'transformer': IndividualTF,
    'singleLSTM': SingleLSTM, 
    'doubleLSTM': DoubleLSTM,
    # 'doubleLSTMGoal': DoubleLSTMGoal,
    'doubleLSTMPure': DoubleLSTMPure,
    'doubleCVAELSTM': DoubleCVAELSTM,
    'CVAEDoubleLSTM': CVAEDoubleLSTM,
    'CVAEMLPLSTM': CVAEMLPLSTM,
    'CVAEDoubleMLP': CVAEDoubleMLP,
    'resnetTraj': resnetTraj,
    'PECNet': PECNet,
    'RVNet': RVNet,
    'RVNetResdual': RVNetResdual,
    'RVNetTransformer': RVNetTransformer,
    'RVNetConv1d': RVNetConv1d,
    'LocalNet': LocalNet
}