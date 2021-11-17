from .YNet import YNetTorch as YNet
from .Transformer import IndividualTF
from .LSTM import SingleLSTM
model_dict = {
    'ynet': YNet,
    'transformer': IndividualTF,
    'singleLSTM': SingleLSTM
}