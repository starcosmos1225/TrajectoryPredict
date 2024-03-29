from .YNet import YNetTorch as YNet
from .YNet import YNetTorchNoGoal as YNetNoGoal
from .Transformer import IndividualTF
from .LSTM import SingleLSTM, DoubleLSTM,DoubleCVAELSTM, DoubleLSTMPure,  \
    CVAEDoubleLSTM, CVAEMLPLSTM
from .resTraj import resnetTraj,CVAEresnetTraj,CVAETeacherTraj,CVAEResMLP
from .PECNet import PECNet
from .NonSquence import CVAEDoubleMLP
from .RVNet import RVNet, RVNetResdual, RVNetTransformer,RVNetConv1d
from .localNet import LocalNet
from .UNet import UNet


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
    'CVAEresnetTraj': CVAEresnetTraj,
    'CVAETeacherTraj': CVAETeacherTraj,
    'CVAEResMLP': CVAEResMLP,
    'PECNet': PECNet,
    'RVNet': RVNet,
    'RVNetResdual': RVNetResdual,
    'RVNetTransformer': RVNetTransformer,
    'RVNetConv1d': RVNetConv1d,
    'LocalNet': LocalNet,
    'UNet': UNet
}