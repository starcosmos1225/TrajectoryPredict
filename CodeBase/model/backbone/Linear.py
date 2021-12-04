from torch import nn
import math

class LinearEmbedding(nn.Module):
    def __init__(self, inp_size,d_model):
        super(LinearEmbedding, self).__init__()
        # lut => lookup table
        self.lut = nn.Linear(inp_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

'''MLP model'''
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        if hasattr(hidden_size,'__iter__'):
            dims.extend(hidden_size)
        else:
            dims.append(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers)-1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x

# ResMLP(inp=num_classes*num_samples*2*pred_len, out=pred_len*2,scale=0.25,activate='ReLU',bn=True)
activate_op = {
    'ReLU': nn.ReLU()
}
class ResMLPBlock(nn.Module):

    def __init__(self, inp, out, middle, activate):
        super(ResMLPBlock, self).__init__()
        self.activate = activate
        self.block1 =  nn.Sequential(
            nn.Linear(inp, middle),
            nn.Linear(middle, inp),
            # nn.BatchNorm1d(inp),
            self.activate
        )
        self.block2 = nn.Linear(inp, out)
        self.bn = nn.BatchNorm1d(out)        

    def forward(self, x):
        res = x
        mid = self.block1(x)
        out = mid + res
        out = self.block2(out)
        # out = self.bn(out)
        out = self.activate(out)
        return out


class ResMLP(nn.Module):
    def __init__(self, inp, out,layers=[1024,512,256,128],scale=0.5, activate="ReLU",bn=True):
        super(ResMLP, self).__init__()
        self.inplane = inp
        self.activate = activate_op[activate]

        self.layerInp = nn.Linear(inp, layers[0]*2)
        self.maxpool = nn.MaxPool1d(2)
        self.inplane = layers[0]
        self.bn1 = nn.BatchNorm1d(layers[0]*2)
        self.layer1 = self._makeLayer(layers[0],scale)
        self.layer2 = self._makeLayer(layers[1],scale)
        self.layer3 = self._makeLayer(layers[2],scale)
        self.layer4 = self._makeLayer(layers[3],scale)
        self.out = nn.Linear(layers[3], out)
    
    def _makeLayer(self, out, scale):
        inp = self.inplane
        self.inplane = out
        middle = int(inp * scale)
        return ResMLPBlock(inp, out, middle, self.activate)

    def forward(self, x):
        # print(x.shape)
        x = self.layerInp(x)
        # print(x.shape)
        # x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = self.out(x)
        return out
