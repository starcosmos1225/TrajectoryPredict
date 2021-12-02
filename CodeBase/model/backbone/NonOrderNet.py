from torch import nn
from .Linear import MLP,LinearEmbedding


class NonOrderNet(nn.Module):
    def __init__(self, inp, out, feat, type='max'):
        super(NonOrderNet, self).__init__()
        self.feat = LinearEmbedding(inp, out)
        self.bottlenec = MLP(out,out,feat)
        if type=='max':
            self.gap = nn.AdaptiveMaxPool2d((1, out))
        elif type=='avg':
            self.gap = nn.AdaptiveAvgPool2d((1, out))
         

    def forward(self, x):
        x = self.feat(x)
        x = self.bottlenec(x)
        x = self.gap(x)
        return x