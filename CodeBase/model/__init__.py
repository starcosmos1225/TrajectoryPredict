from .YNet import YNetTorch as YNet
from .Transformer import IndividualTF
model_dict = {
    'ynet': YNet,
    'transformer': IndividualTF,
}