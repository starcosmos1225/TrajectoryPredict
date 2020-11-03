import torch
import torch.nn as nn
from torch.nn import Parameter
#import torch.jit as jit
import warnings
from collections import namedtuple
from typing import List, Tuple
from torch import Tensor
import numbers
import math
import torch.nn.modules.rnn as rnn
from  torch.nn.utils.rnn import PackedSequence
from typing import List, Tuple, Optional, overload
DEVICE = 'cpu'
'''
Perfome a SCF+GRU structure
'''
# class GRUCell(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(GRUCell, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.weight_ir = Parameter(torch.randn( input_size))
#         self.weight_hr = Parameter(torch.randn(hidden_size))
#         self.bias_ir = Parameter(torch.randn(1))
#         self.bias_hr = Parameter(torch.randn(1))
#         self.weight_iz = Parameter(torch.randn(input_size))
#         self.weight_hz = Parameter(torch.randn(hidden_size))
#         self.bias_iz = Parameter(torch.randn(1))
#         self.bias_hz = Parameter(torch.randn(1))
#         self.weight_in = Parameter(torch.randn(input_size))
#         self.weight_hn = Parameter(torch.randn(hidden_size))
#         self.bias_in = Parameter(torch.randn(1))
#         self.bias_hn = Parameter(torch.randn(1))

#     #@jit.script_method
#     def forward(self, input_x, hidden):
#         # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
#         rt = torch.sigmoid(torch.mm(input_x,self.weight_ir)+self.bias_ir+torch.mm(hidden,self.weight_hr)+self.bias_hr)
#         zt = torch.sigmoid(torch.mm(input_x,self.weight_iz)+self.bias_iz+torch.mm(hidden,self.weight_hz)+self.bias_hz)
#         nt = torch.tanh(torch.mm(input_x,self.weight_in)+self.bias_in+rt*(torch.mm(hidden,self.weight_hn)+self.bias_hn))
#         ht = (1-zt)*nt + zt*hidden
#         return ht

# class SCF_GRUCell(nn.Module):
#     def __init__(self, input_size, hidden_size,radius_range,social_pooling_size):
#         super(SCF_GRUCell, self).__init__()
#         self.radius_range = radius_range
#         self.social_pooling_size = social_pooling_size
#         self.radius_step = (self.radius_range[1]-self.radius_range[0])/social_pooling_size[0]
#         self.theta_step = 2*math.pi/social_pooling_size[1]
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.weight_ir = Parameter(torch.randn( input_size))
#         self.weight_hr = Parameter(torch.randn(hidden_size))
#         self.bias_ir = Parameter(torch.randn(1))
#         self.bias_hr = Parameter(torch.randn(1))
#         self.weight_iz = Parameter(torch.randn(input_size))
#         self.weight_hz = Parameter(torch.randn(hidden_size))
#         self.bias_iz = Parameter(torch.randn(1))
#         self.bias_hz = Parameter(torch.randn(1))
#         self.weight_in = Parameter(torch.randn(input_size))
#         self.weight_hn = Parameter(torch.randn(hidden_size))
#         self.bias_in = Parameter(torch.randn(1))
#         self.bias_hn = Parameter(torch.randn(1))
#         self.fc = nn.Sequential(nn.Linear(self.social_pooling_size[0]*self.social_pooling_size[1]*self.hidden_size,self.hidden_size),
#                                 nn.ReLU())
                            
#     def compute_dist(self,loc_a,loc_b):
#       '''
#       loc_a:tensor(2)
#       loc_b:tensor(2)
#       return: distance (a-b) tensor(1) float
#       '''
#       return torch.norm(loc_a-loc_b)
    
#     def compute_theta(self,loc_a,loc_b):
#       '''
#       loc_a:tensor(2)
#       loc_b:tensor(2)
#       return: angle (b-a) tensor(1) float(0~2pi)
#       '''
#       c = loc_b-loc_a
#       dist = torch.norm(c)
#       costheta = c[0]/dist
#       if (c[1]<0):
#         theta = 2*math.pi - costheta.acos()
#       else:
#         theta = costheta.acos()
#       return theta
#     #@jit.script_method
#     def forward(self, loc_agent,loc_others,loc_other_index,feature_img,f_vel, hiddens,hidden):
#         # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
#         '''
#         loca_agent:tensor(2)
#         loc_others:tensor(n-1,2)
#         loc_other_index:list of other agents' index
#         feature_img: tensor(32,80,80)
#         f_vel:tensor(16)
#         hiddens:tensor(n,48)
#         hidden:tensor(48)
#         '''
#         H = feature_img.shape[1]
#         W = feature_img.shape[2]
#         nums_feature = feature_img.shape[1]
#         u = int(H/2-int(loc_agent[1]))
#         v = int(loc_agent[0])
#         # feature_agent:(32)
#         feature_agent = feature_img[ :, u, v]
#         # sp: tensor(6,6,48)
#         sp = torch.zeros((self.social_pooling_size[0],self.social_pooling_size[1],self.hidden_size), device=torch.device(DEVICE))
#         # sp_c: count the numbers in (6,6)
#         sp_c = torch.zeros((self.social_pooling_size[0],self.social_pooling_size[1]), device=torch.device(DEVICE))
#         # print(loc_others.shape[0])
#         # print("len:{}".format(len(loc_other_index)))
#         # t=input
#         for i in range(loc_others.shape[0]):
#           # loc:tensor(2)
#           loc = loc_others[i]
#           # dist:tensor(1)
#           dist = self.compute_dist(loc, loc_agent)
#           if self.radius_range[0] <= dist <= self.radius_range[1]:
#             theta = self.compute_theta(loc_agent, loc)
#             u = int((dist-self.radius_range[0])//self.radius_step)
#             v = int((theta//self.theta_step))
#             sp[u,v] += hiddens[loc_other_index[i]]
#             sp_c[u,v] += 1
#         for i in range(self.social_pooling_size[0]):
#           for j in range(self.social_pooling_size[1]):
#             if sp_c[i][j] > 1.0:
#               sp[i][j] = sp[i][j]/sp_c[i][j]
#         #(6,6,48)->(6*6*48)
#         sp = sp.view(self.social_pooling_size[0]*self.social_pooling_size[1]*self.hidden_size)
#         #(6*6*48)->(48)
#         fsp = self.fc(sp)
#         input_x = torch.cat((feature_agent,f_vel,fsp),0)
#         #(32)+(16)+(48)=(96)
#         assert input_x.shape[0]==96
#         rt = torch.sigmoid(torch.sum(torch.mul(input_x,self.weight_ir))+self.bias_ir+torch.sum(torch.mul(hidden,self.weight_hr))+self.bias_hr)
#         zt = torch.sigmoid(torch.sum(torch.mul(input_x,self.weight_iz))+self.bias_iz+torch.sum(torch.mul(hidden,self.weight_hz))+self.bias_hz)
#         nt = torch.tanh(torch.sum(torch.mul(input_x,self.weight_in))+self.bias_in+rt*(torch.sum(torch.mul(hidden,self.weight_hn))+self.bias_hn))
#         ht = (1-zt)*nt + zt*hidden
#         return ht


# class GRULayer(nn.Module):
#     def __init__(self, cell, numbers_layers,*cell_args):
#         super(GRULayer, self).__init__()
#         self.cell = cell(*cell_args)

#     #@jit.script_method
#     def forward(self, input, state):
#         # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
#         inputs = input.unbind(0)
#         outputs = []
#         for i in range(len(inputs)):
#             state = self.cell(inputs[i], state)
#             outputs += [state]
#         return torch.stack(outputs), state

# class SCF_GRULayer(nn.Module):
#     def __init__(self, cell, batch_size, nums_sample,numbers_layers,*cell_args):
#         super(SCF_GRULayer, self).__init__()
#         self.cell = cell(*cell_args)
#         self.cell.to(DEVICE)
#         self.K = nums_sample
#         self.numbers_layers = numbers_layers
#         self.batch_size = batch_size

#     #@jit.script_method
#     def forward(self,path,f_vel,f_img):
#         # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
#         '''
#         path:tensor(40,batch_size*n,2)
#         f_vel:tensor(40,batch_size*n,16)
#         f_img:tensor(batch_size,32,80,80)
#         return output_x(40,batch_size*n,48),hidden_n(batch_size*n,48)
#         '''
#         assert path.shape[0]==40
#         nums_agent = int(path.shape[1]/self.batch_size)
#         outputs = []
#         #state:tensor(n,hidden)
#         state = torch.zeros((path.shape[1], self.cell.hidden_size), device=torch.device(DEVICE))
#         # print(f_img.shape)
#         # print("batch_size:{}".format(self.batch_size))
#         for i in range(self.numbers_layers):
#           new_state = state.clone()
#           for k in range(self.batch_size):
#               for j in range(nums_agent):
#                 # tensor(2)
#                 loc_agent = path[i][k*nums_agent+j]
#                 # tensor(nums_agent,2)
#                 loc_other = torch.zeros((nums_agent-1, 2), device=torch.device(DEVICE))
#                 loc_other_index = []
#                 index = 0
#                 for t in range(nums_agent):
#                   if t != j:
#                     loc_other[index] = path[i][k*nums_agent+t]
#                     loc_other_index.append(t)
#                     index += 1
#                 new_state[k*nums_agent+j] = self.cell(loc_agent, loc_other,
#                                                       loc_other_index, f_img[k],
#                                                       f_vel[i][k*nums_agent+j], state[i:i+nums_agent], state[k*nums_agent+j])
#           state = new_state.clone()
#           outputs += [state]
#         return torch.stack(outputs), state

# class GRU(nn.Module):
#     def __init__(self, input_size,hidden_size,nums_layers):
#         super(GRU, self).__init__()
#         self.layer = GRULayer(GRUCell,nums_layers,input_size,hidden_size)

#     #@jit.script_method
#     def forward(self, input, state):
#         return self.layer.forward(input,state)

# class SCF_GRU(nn.Module):
#     def __init__(self, batch_size, nums_sample,input_size,hidden_size,nums_layers,radius_range,social_pooling_size,device):
#         global DEVICE
#         super(SCF_GRU, self).__init__()
#         DEVICE=device
#         self.layer = SCF_GRULayer(SCF_GRUCell, batch_size, nums_sample,nums_layers,input_size,hidden_size,radius_range,social_pooling_size)
#         self.layer.to(DEVICE)
#     #@jit.script_method
#     def forward(self, Y_path,Y_fv,feature_map):
#       '''
#       Y_path:Tensor (40,n,2)
#       Y_fv: Tensor (40,n,16)
#       feature_map: Tensor (batch_size,32,80,80)
#       '''
#       return self.layer.forward(Y_path,Y_fv,feature_map)


# class GRU_TEST(rnn.RNNBase):
#     r"""Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.


#     For each element in the input sequence, each layer computes the following
#     function:

#     .. math::
#         \begin{array}{ll}
#             r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
#             z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
#             n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
#             h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
#         \end{array}

#     where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the input
#     at time `t`, :math:`h_{(t-1)}` is the hidden state of the layer
#     at time `t-1` or the initial hidden state at time `0`, and :math:`r_t`,
#     :math:`z_t`, :math:`n_t` are the reset, update, and new gates, respectively.
#     :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product.

#     In a multilayer GRU, the input :math:`x^{(l)}_t` of the :math:`l` -th layer
#     (:math:`l >= 2`) is the hidden state :math:`h^{(l-1)}_t` of the previous layer multiplied by
#     dropout :math:`\delta^{(l-1)}_t` where each :math:`\delta^{(l-1)}_t` is a Bernoulli random
#     variable which is :math:`0` with probability :attr:`dropout`.

#     Args:
#         input_size: The number of expected features in the input `x`
#         hidden_size: The number of features in the hidden state `h`
#         num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
#             would mean stacking two GRUs together to form a `stacked GRU`,
#             with the second GRU taking in outputs of the first GRU and
#             computing the final results. Default: 1
#         bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
#             Default: ``True``
#         batch_first: If ``True``, then the input and output tensors are provided
#             as (batch, seq, feature). Default: ``False``
#         dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
#             GRU layer except the last layer, with dropout probability equal to
#             :attr:`dropout`. Default: 0
#         bidirectional: If ``True``, becomes a bidirectional GRU. Default: ``False``

#     Inputs: input, h_0
#         - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
#           of the input sequence. The input can also be a packed variable length
#           sequence. See :func:`torch.nn.utils.rnn.pack_padded_sequence`
#           for details.
#         - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
#           containing the initial hidden state for each element in the batch.
#           Defaults to zero if not provided. If the RNN is bidirectional,
#           num_directions should be 2, else it should be 1.

#     Outputs: output, h_n
#         - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
#           containing the output features h_t from the last layer of the GRU,
#           for each `t`. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
#           given as the input, the output will also be a packed sequence.
#           For the unpacked case, the directions can be separated
#           using ``output.view(seq_len, batch, num_directions, hidden_size)``,
#           with forward and backward being direction `0` and `1` respectively.

#           Similarly, the directions can be separated in the packed case.
#         - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
#           containing the hidden state for `t = seq_len`

#           Like *output*, the layers can be separated using
#           ``h_n.view(num_layers, num_directions, batch, hidden_size)``.

#     Shape:
#         - Input1: :math:`(L, N, H_{in})` tensor containing input features where
#           :math:`H_{in}=\text{input\_size}` and `L` represents a sequence length.
#         - Input2: :math:`(S, N, H_{out})` tensor
#           containing the initial hidden state for each element in the batch.
#           :math:`H_{out}=\text{hidden\_size}`
#           Defaults to zero if not provided. where :math:`S=\text{num\_layers} * \text{num\_directions}`
#           If the RNN is bidirectional, num_directions should be 2, else it should be 1.
#         - Output1: :math:`(L, N, H_{all})` where :math:`H_{all}=\text{num\_directions} * \text{hidden\_size}`
#         - Output2: :math:`(S, N, H_{out})` tensor containing the next hidden state
#           for each element in the batch

#     Attributes:
#         weight_ih_l[k] : the learnable input-hidden weights of the :math:`\text{k}^{th}` layer
#             (W_ir|W_iz|W_in), of shape `(3*hidden_size, input_size)` for `k = 0`.
#             Otherwise, the shape is `(3*hidden_size, num_directions * hidden_size)`
#         weight_hh_l[k] : the learnable hidden-hidden weights of the :math:`\text{k}^{th}` layer
#             (W_hr|W_hz|W_hn), of shape `(3*hidden_size, hidden_size)`
#         bias_ih_l[k] : the learnable input-hidden bias of the :math:`\text{k}^{th}` layer
#             (b_ir|b_iz|b_in), of shape `(3*hidden_size)`
#         bias_hh_l[k] : the learnable hidden-hidden bias of the :math:`\text{k}^{th}` layer
#             (b_hr|b_hz|b_hn), of shape `(3*hidden_size)`

#     .. note::
#         All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
#         where :math:`k = \frac{1}{\text{hidden\_size}}`

#     .. include:: ../cudnn_persistent_rnn.rst

#     Examples::

#         >>> rnn = nn.GRU(10, 20, 2)
#         >>> input = torch.randn(5, 3, 10)
#         >>> h0 = torch.randn(2, 3, 20)
#         >>> output, hn = rnn(input, h0)
#     """

#     def __init__(self, *args, **kwargs):
#         super(GRU_TEST, self).__init__('GRU', *args, **kwargs)

#     @overload
#     @torch._jit_internal._overload_method  # noqa: F811
#     def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:  # noqa: F811
#         pass

#     @overload
#     @torch._jit_internal._overload_method  # noqa: F811
#     def forward(self, input: PackedSequence, hx: Optional[Tensor] = None) -> Tuple[PackedSequence, Tensor]:  # noqa: F811
#         pass

#     def forward(self, input, hx=None):  # noqa: F811
#         orig_input = input
#         # xxx: isinstance check needs to be in conditional for TorchScript to compile
#         if isinstance(orig_input, PackedSequence):
#             input, batch_sizes, sorted_indices, unsorted_indices = input
#             max_batch_size = batch_sizes[0]
#             max_batch_size = int(max_batch_size)
#         else:
#             batch_sizes = None
#             max_batch_size = input.size(0) if self.batch_first else input.size(1)
#             sorted_indices = None
#             unsorted_indices = None

#         if hx is None:
#             num_directions = 2 if self.bidirectional else 1
#             hx = torch.zeros(self.num_layers * num_directions,
#                              max_batch_size, self.hidden_size,
#                              dtype=input.dtype, device=input.device)
#         else:
#             # Each batch of the hidden state should match the input sequence that
#             # the user believes he/she is passing in.
#             hx = self.permute_hidden(hx, sorted_indices)

#         self.check_forward_args(input, hx, batch_sizes)
#         if batch_sizes is None:
#             result = rnn._VF.gru(input, hx, self._flat_weights, self.bias, self.num_layers,
#                              self.dropout, self.training, self.bidirectional, self.batch_first)
#         else:
#             result = rnn._VF.gru(input, batch_sizes, hx, self._flat_weights, self.bias,
#                              self.num_layers, self.dropout, self.training, self.bidirectional)
#         output = result[0]
#         hidden = result[1]

#         # xxx: isinstance check needs to be in conditional for TorchScript to compile
#         if isinstance(orig_input, PackedSequence):
#             output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
#             return output_packed, self.permute_hidden(hidden, unsorted_indices)
#         else:
#             return output, self.permute_hidden(hidden, unsorted_indices)

