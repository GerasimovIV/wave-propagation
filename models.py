import torch, torch.nn as nn
import torch.nn.functional as F
#from torch.nn.parameter import Parameter

from conv_gru import ConvGRU, ConvGRUCell
from conv_lstm import ConvLSTM, ConvLSTMCell
from losses import WaveLoss, hessian_diag_u

def sparseFunction(x, s, activation=torch.relu, f=torch.sigmoid):
  w = torch.abs(x)
  return torch.sign(x) * activation(w - f(s))


class STR(nn.Module):
  def __init__(self):
    super(STR, self).__init__()
    self.activation = torch.relu
    self.threshold = nn.Parameter(torch.Tensor([1]), requires_grad=True)
    self.threshold.data.uniform_(-6., -3.5)
    self.f = torch.sigmoid

  def to(self, device):
    data = self.threshold.data.to(device)
    self.threshold = nn.Parameter(data, requires_grad=True)


  def get_sparsity(self, f=torch.sigmoid):
    sparse_weights = sparseFunction(self.weight, self.threshold,  self.activation, self.f)
    temp = sparse_weights.detach().cpu()
    temp[temp != 0] = 1
    return (100 - temp.mean() * 100), temp.numel(), f(self.threshold)


class STRConv(nn.Conv2d, STR):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    #print(1)


  def forward(self, x):
      sparse_weights = sparseFunction(self.weight, self.threshold, self.activation, self.f)
      x = F.conv2d(x, sparse_weights, self.bias, self.stride, self.padding, self.dilation, self.groups)
      return x


class STRConvTranspose(nn.ConvTranspose2d, STR):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)   

  def forward(self, x):
      sparse_weights = sparseFunction(self.weight, self.threshold, self.activation, self.f)
      x = F.conv_transpose2d(x, sparse_weights, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)
      return x


def conv_block(in_channels, out_channels, kernel_size, stride, padding, 
                pruning, pooling, activation):
  
  conv_2d = nn.Conv2d if not pruning else STRConv

  return nn.Sequential(conv_2d(in_channels, out_channels, kernel_size, padding=padding), 
                      #  nn.BatchNorm2d(out_channels),
                      #  activation(),
                      #  conv_2d(out_channels, out_channels, kernel_size, padding=padding), 
                       pooling(2),
                       activation())


class STRConvGRUCell(ConvGRUCell, nn.Module):
  def __init__(self, input_dim, hidden_dim, kernel_size, bias, dtype, pruning=True):
    #super(STRConvGRUCell, self).__init__(input_dim, hidden_dim, kernel_size, bias, dtype)
    super(ConvGRUCell, self).__init__()

    self.padding = kernel_size[0] // 2, kernel_size[1] // 2
    self.hidden_dim = hidden_dim
    self.bias = bias
    self.dtype = dtype

    conv_class = STRConv if pruning else nn.Conv2d

    self.conv_gates = conv_class(in_channels=input_dim + hidden_dim,
                                out_channels=2*self.hidden_dim,  # for update_gate,reset_gate respectively
                                kernel_size=kernel_size,
                                padding=self.padding,
                                bias=self.bias)

    self.conv_can = conv_class(in_channels=input_dim + hidden_dim,
                          out_channels=self.hidden_dim, # for candidate neural memory
                          kernel_size=kernel_size,
                          padding=self.padding,
                          bias=self.bias)


class STRConvLSTMCell(ConvLSTMCell, nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, bias, dtype, pruning=True):

        super(ConvLSTMCell, self).__init__()

        self.input_dim  = in_channels
        self.hidden_dim = hidden_channels

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        conv_class = STRConv if pruning else nn.Conv2d
        
        self.conv = conv_class(in_channels=self.input_dim + self.hidden_dim,
                                  out_channels=4 * self.hidden_dim,
                                  kernel_size=self.kernel_size,
                                  padding=self.padding,
                                  bias=self.bias)


class FastConvGRUCell(ConvGRUCell, nn.Module):
  def __init__(self, input_dim, hidden_dim, kernel_size, bias, dtype, pruning=True):
    #super(FastConvGRUCell, self).__init__(input_dim, hidden_dim, kernel_size, bias, dtype)
    super(ConvGRUCell, self).__init__()

    self.zeta_nu = nn.Parameter(torch.Tensor([1., -4.]), requires_grad=True)
    nn.init.uniform_(self.zeta_nu, -6., -2.)

    self.f = torch.sigmoid

    self.padding = kernel_size[0] // 2, kernel_size[1] // 2
    self.hidden_dim = hidden_dim
    self.bias = bias
    self.dtype = dtype

    conv_class = STRConv if pruning else nn.Conv2d
    #print(conv_class)
    self.conv_gates = conv_class(in_channels=input_dim + hidden_dim,
                                 out_channels=self.hidden_dim,  # for update_gate,reset_gate respectively
                                 kernel_size=kernel_size,
                                 padding=self.padding,
                                 bias=self.bias)
    
    self.conv_can = conv_class(in_channels=input_dim + hidden_dim,
                          out_channels=self.hidden_dim, # for candidate neural memory
                          kernel_size=kernel_size,
                          padding=self.padding,
                          bias=self.bias)

    self.conv_can.weight = self.conv_gates.weight
    

  def forward(self, input_tensor, h_cur):
    combined = torch.cat([input_tensor, h_cur], dim=1)
    combined_conv = self.conv_gates(combined)

    update_gate = torch.sigmoid(combined_conv)

    cc_cnm = self.conv_can(combined)
    cnm = torch.tanh(cc_cnm)
    
    zeta_nu = self.f(self.zeta_nu)

    h_next = (zeta_nu[0] * (1 - update_gate) + zeta_nu[1]) * cnm + update_gate * h_cur
    return h_next


class STRConvGRU(ConvGRU, nn.Module):
  def __init__(self, 
               input_dim, hidden_dim, kernel_size, num_layers,
               dtype, batch_first=False, bias=True, return_all_layers=False, 
               mode='fast', pruning=True):
    
    super(ConvGRU, self).__init__()
    
    kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
    hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
    
    if not len(kernel_size) == len(hidden_dim) == num_layers:
        raise ValueError('Inconsistent list length.')

    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.kernel_size = kernel_size
    self.dtype = dtype
    self.num_layers = num_layers
    self.batch_first = batch_first
    self.bias = bias
    self.return_all_layers = return_all_layers

    cell = STRConvGRUCell if mode != 'fast' else FastConvGRUCell
    #print(cell)
    cell_list = []
    for i in range(0, self.num_layers):
        cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
        cell_list.append(cell(#input_size=(self.height, self.width),
                              input_dim=cur_input_dim,
                              hidden_dim=self.hidden_dim[i],
                              kernel_size=self.kernel_size[i],
                              bias=self.bias,
                              dtype=self.dtype,
                              pruning=pruning))

    self.cell_list = nn.ModuleList(cell_list)


class STRConvLSTM(ConvLSTM, nn.Module):
  def __init__(self, 
               in_channels, hidden_channels, kernel_size, num_layers,
               dtype=torch.cuda.FloatTensor, batch_first=False, bias=True, return_all_layers=False, 
               mode='fast', pruning=True):
    
    super(ConvLSTM, self).__init__()
    
    kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
    hidden_channels  = self._extend_for_multilayer(hidden_channels, num_layers)
    
    if not len(kernel_size) == len(hidden_channels) == num_layers:
        raise ValueError('Inconsistent list length.')

    self.input_dim = in_channels
    self.hidden_dim = hidden_channels
    self.kernel_size = kernel_size
    self.dtype = dtype
    self.num_layers = num_layers
    self.batch_first = batch_first
    self.bias = bias
    self.return_all_layers = return_all_layers

    cell = STRConvLSTMCell# if mode != 'fast' else FastConvGRUCell
    #print(cell)
    cell_list = []
    for i in range(0, self.num_layers):
        cur_input_dim = in_channels if i == 0 else hidden_channels[i - 1]
        cell_list.append(cell(#input_size=(self.height, self.width),
                              in_channels=cur_input_dim,
                              hidden_channels=self.hidden_dim[i],
                              kernel_size=self.kernel_size[i],
                              bias=self.bias,
                              dtype=self.dtype,
                              pruning=pruning))

    self.cell_list = nn.ModuleList(cell_list)


class Encoder(nn.Module):
  def __init__(self, 
              in_channels=1, out_channels=32, kernel_size=3, padding=1, n_layers=2,
              pruning=True, pooling=nn.MaxPool2d, activation=nn.Softplus):
    super(Encoder, self).__init__()
    
    self.model = nn.Sequential()
    for i in range(n_layers):  
      self.model.add_module('conv_block_' + str(i), 
                            conv_block(in_channels, out_channels // (n_layers) * (i + 1), kernel_size, 1,
                                       padding, pruning, pooling, activation))
      in_channels = out_channels // (n_layers) * (i + 1)
 
  
  def forward(self, input):
    return self.model(input)


def conv_T_block(in_channels, out_channels, kernel_size, stride, padding,
                 pruning, activation):
  
  conv_T_2d = nn.ConvTranspose2d if not pruning else STRConvTranspose

  return nn.Sequential(nn.Upsample(scale_factor=2., mode='bilinear', align_corners=False),
                      #  nn.BatchNorm2d(in_channels),
                       activation(),
                       conv_T_2d(in_channels, out_channels, kernel_size, stride, padding),
                      #  activation(),
                      #  conv_T_2d(out_channels, out_channels, kernel_size, stride, padding)
                      )


class Decoder(nn.Module):
  def __init__(self, in_channels=32, out_channels=1, kernel_size=3, padding=1, stride=1, n_layers=2, 
               pruning=True, activation=nn.Softplus):
    super(Decoder, self).__init__()
    self.out_channels = out_channels
    
    self.model = nn.Sequential()
    for i in range(n_layers):
      #out = in_channels // 2 if i != n_layers - 1 and self.out_channels != 1 else 1
      out = in_channels // 2 if self.out_channels != 1 else 1
      
      self.model.add_module('conv_T_block_' + str(i), conv_T_block(in_channels, out, kernel_size, stride, padding, pruning, activation))
      #if i != n_layers - 1:
      if self.out_channels != 1:
        self.model.add_module('activ_' + str(i), activation())

      in_channels = in_channels // 2

  
  def forward(self, input):
    return self.model(input)


class Decoder_old(nn.Module):
  def __init__(self, in_channels=32, out_channels=1, kernel_size=3, padding=1, stride=1, n_layers=2, 
               pruning=True, activation=nn.Softplus):
    super(Decoder_old, self).__init__()
    
    self.model = nn.Sequential()
    for i in range(n_layers):
      out = in_channels // 2 if i != n_layers - 1 else 1
      
      self.model.add_module('conv_T_block_' + str(i), conv_T_block(in_channels, out, kernel_size, stride, padding, pruning))
      if i != n_layers - 1:
        self.model.add_module('activ_' + str(i), activation())

      in_channels = in_channels // 2

  
  def forward(self, input):
    return self.model(input)


def seq_to_cnn(data):
  shape = data.shape
  return data.view(shape[0] * shape[1], *shape[2:])


class BaseWave(nn.Module):
  
  def compression(self):
    for i in self.modules():
      if type(i) in [STRConv, STRConvTranspose]:
        print(i.get_sparsity())


class WaveAE(BaseWave):
  def __init__(self, in_channels=2,
               bottle_neck=32, kernel_size=3, pruning=False, n_layers=2,
               pooling=nn.MaxPool2d, activation=nn.Softplus):
    super(WaveAE, self).__init__()
    
    self.model_type = 'AE'
    self.pruning = pruning
    self.encoder = Encoder(in_channels=in_channels, out_channels=bottle_neck, n_layers=n_layers,
                           kernel_size=kernel_size, pruning=pruning, pooling=pooling, activation=activation)
    
    self.decoder = Decoder_old(in_channels=bottle_neck, out_channels=1, n_layers=n_layers, stride=1,
                           kernel_size=kernel_size, pruning=pruning, activation=activation)


  def forward(self, x):
    shape = x.shape
    if len(x.shape) == 4:
      output = x.unsqueeze(2)
    else:
      output = x
    
    output = self.encoder(seq_to_cnn(output))
    output = self.decoder(output)

    return output.view(shape[0], shape[1], 1, x.shape[-2], x.shape[-1]).squeeze(2)
  
  # def loss(self, X, y, device='cuda'):
  #   predictions = self.forward(X.to(device))
  #   return nn.L1Loss(X, y.to(device))



# class WaveVAE(BaseWave):
#   def __init__(self, in_channels=2, bottle_neck=32, kernel_size=3, pruning=False, n_layers=2, 
#                pooling=nn.MaxPool2d, activation=nn.Softplus):
#     super(WaveVAE, self).__init__()
    
#     self.model_type = 'AE'

#     self.encoder = Encoder(in_channels=1, out_channels=bottle_neck, n_layers=n_layers,
#                            kernel_size=kernel_size, pruning=pruning, pooling=pooling, activation=activation)
    
#     self.decoder = Decoder(in_channels=bottle_neck, out_channels=1, n_layers=n_layers,
#                            kernel_size=kernel_size, pruning=pruning, activation=activation)

  
#   def reparameterize(self, mu, log_var):
#     std = torch.exp(0.5 * log_var) # standard deviation
#     eps = torch.randn_like(std) # `randn_like` as we need the same size
#     sample = mu + (eps * std) # sampling
#     return sample
  
  
#   def forward(self, x):
#     shape = x.shape
#     x = self.encoder(seq_to_cnn(x))
#     mu, log_var = x, x
    
#     x = self.reparameterize(mu, log_var)
#     x = self.decoder(x)
#     x = torch.sigmoid(x)

#     x = x.view(shape[0], shape[1], x.shape[1], x.shape[2], x.shape[3]).squeeze(2)
#     mu = mu.view(shape[0], shape[1], mu.shape[1], mu.shape[2], mu.shape[3]).squeeze(2)
#     log_var = log_var.view(shape[0], shape[1], log_var.shape[1], log_var.shape[2], log_var.shape[3]).squeeze(2)
#     return x, mu, log_var


class WaveGRUModel(BaseWave):
  def __init__(self, 
               bottle_neck=32, 
               rnn_channels=32,
               n_layers=2,
               kernel_size=3,
               pooling=nn.MaxPool2d,
               activation=nn.Softplus, 
               mode='fast', 
               pruning=True, 
               hidden_control=None):
    super(WaveGRUModel, self).__init__()
    
    self.model_type = 'GRU'
    self.pruning = pruning

    if hidden_control is not None:
      assert isinstance(hidden_control, dict)
      assert 'ord' in hidden_control.keys()
      assert 'lam' in hidden_control.keys()

      self.hidden_loss = WaveLoss(order=hidden_control['ord'], scale=hidden_control['lam'])

    
    self.hidden_control = hidden_control

    self.data_to_h0 = Encoder(in_channels=1, out_channels=bottle_neck, n_layers=n_layers,
                           kernel_size=kernel_size, pruning=pruning, pooling=pooling, activation=activation)
    self.encoder = Encoder(in_channels=1, out_channels=bottle_neck // 2, n_layers=n_layers // 2,
                           kernel_size=kernel_size, pruning=pruning, pooling=pooling, activation=activation)

    self.encoder_1 = Encoder(in_channels=bottle_neck // 2, out_channels=bottle_neck, n_layers=n_layers // 2,
                           kernel_size=kernel_size, pruning=pruning, pooling=pooling, activation=activation)

    self.decoder  = Decoder(in_channels=rnn_channels, out_channels=rnn_channels // 2, n_layers=n_layers // 2, stride=1,
                           kernel_size=kernel_size, pruning=pruning, activation=activation)
    
    self.decoder_1  = Decoder(in_channels=rnn_channels // 2, out_channels=1, n_layers=n_layers // 2, stride=1,
                           kernel_size=kernel_size, pruning=pruning, activation=activation)
    
    self.conv_gru = STRConvGRU(input_dim=bottle_neck,
                              hidden_dim=[rnn_channels],
                              kernel_size=(kernel_size, kernel_size),
                              dtype='str',
                              num_layers=1,
                              batch_first=True,
                              bias = True,
                              return_all_layers = False, pruning=pruning)

  
  def seq_to_cnn(self, data):
    shape = data.shape
    return data.reshape(shape[0] * shape[1], *shape[2:])

  
  def forward(self, solutions, eq_features, *wargs):

    initial_hid = self.data_to_h0(eq_features)
    shape = solutions.shape
    #print(shape)
    out = self.encoder(seq_to_cnn(solutions.contiguous().unsqueeze(2)))
    #print(out.shape)
    x = self.encoder_1(out)
    #print(x.shape)
    x = x.view(shape[0], shape[1], x.shape[1], x.shape[2], x.shape[3])
    #print(x.shape)
    #print(initial_hid.shape)

    output, _ = self.conv_gru(x, initial_hid)#torch.zeros_like(x[:, 0]))
    
    if self.hidden_control is not None:
      h_diff = 0.#self.hidden_loss(output[0][:, :-1, :, :, :], x[:, 1:, :, :, :])
      #h_diff = output[0][:, :-1, :, :, :] - x[:, 1:, :, :, :]
    else:
      h_diff = 0.

    x = output[0]
    #print('here', x.shape)
    #print(seq_to_cnn(x).shape)
    x = self.decoder(seq_to_cnn(x))
    #print('feeee', torch.cat([x, out], dim=1).shape)
    #x = self.decoder_1(torch.cat([x, out], dim=1))
    #x = x + out
    x = self.decoder_1(x) 
    
    #print(x.shape)
    # print(x.shape)
    x = x.view(shape[0], shape[1], x.shape[1], x.shape[2], x.shape[3]).squeeze(2)
    #print(x.shape)
    return x, h_diff, None 


class OneByOne(nn.Module):
  def __init__(self, fc_dims: list = [1, 7], activation=nn.Sigmoid, lam=1e-2):
    super(OneByOne, self).__init__()
    self.model_type = 'OBO'
    self.activation = activation
    self.lam = lam
    assert len(fc_dims) > 2, 'Idiot'

    self.FCs = nn.Sequential()
    for i in range(len(fc_dims) - 1):
      self.FCs.add_module('fc_' + str(i + 1), nn.Linear(fc_dims[i], fc_dims[i + 1]))
      self.FCs.add_module(self.activation.__name__ + '_' + str(i + 1), self.activation())

  def forward(self, X):
    P = self.FCs(X)
    residual = hessian_diag_u(X, self.FCs)
    residual = self.lam * abs((residual[:, 0] + residual[:, 1])*X[:, 3]**2 + X[:, -3]*X[:, -1]**2/X[:, -2]**2 - residual[:, 2])
    return P, residual


class WaveLSTMModel(BaseWave):
  def __init__(self, 
               bottle_neck=32, 
               rnn_channels=32,
               n_layers=2,
               kernel_size=3,
               pooling=nn.MaxPool2d,
               activation=nn.Softplus, 
               mode='fast', 
               pruning=True, 
               hidden_control=None):
    super(WaveLSTMModel, self).__init__()
    
    self.model_type = 'LSTM'
    self.pruning = pruning

    if hidden_control is not None:
      assert isinstance(hidden_control, dict)
      assert 'ord' in hidden_control.keys()
      assert 'lam' in hidden_control.keys()

      self.hidden_loss = WaveLoss(order=hidden_control['ord'], scale=hidden_control['lam'])

    
    self.hidden_control = hidden_control
    
    self.q_diff = nn.Sequential(nn.Linear(2, bottle_neck),
                                activation(),
                                nn.Linear(bottle_neck, 1))

    self.data_to_h0 = Encoder(in_channels=1, out_channels=bottle_neck, n_layers=n_layers,
                           kernel_size=kernel_size, pruning=pruning, pooling=pooling, activation=activation)
    self.encoder = Encoder(in_channels=1, out_channels=bottle_neck // 2, n_layers=n_layers // 2,
                           kernel_size=kernel_size, pruning=pruning, pooling=pooling, activation=activation)

    self.encoder_1 = Encoder(in_channels=bottle_neck // 2, out_channels=bottle_neck, n_layers=n_layers // 2,
                           kernel_size=kernel_size, pruning=pruning, pooling=pooling, activation=activation)

    self.decoder  = Decoder(in_channels=rnn_channels, out_channels=rnn_channels // 2, n_layers=n_layers // 2, stride=1,
                           kernel_size=kernel_size, pruning=pruning, activation=activation)
    
    self.decoder_1  = Decoder(in_channels=rnn_channels // 2, out_channels=1, n_layers=n_layers // 2, stride=1,
                           kernel_size=kernel_size, pruning=pruning, activation=activation)
    # in_channels, hidden_channels, kernel_size, num_layers
    self.conv_lstm = STRConvLSTM(in_channels=bottle_neck,
                              hidden_channels=[rnn_channels],
                              kernel_size=(kernel_size, kernel_size),
                              num_layers=1,
                              dtype=torch.DoubleTensor,
                              batch_first=True,
                              bias = True,
                              return_all_layers = False, 
                              mode=mode, pruning=pruning)

  
  def seq_to_cnn(self, data):
    shape = data.shape
    return data.reshape(shape[0] * shape[1], *shape[2:])

  
  def forward(self, solutions, eq_features, q_prev_curr=None, scri_l=None, scrj_l=None, 
              context=None, forcing=False):
    # print(solutions.shape, eq_features.shape)
    # print(q_prev_curr.shape)
    
    q = q_prev_curr# if forcing else q_prev_curr[:, 0, :].unsqueeze(1)
    
    q_shape = q_prev_curr.shape[:-1]# if forcing else (q_prev_curr.shape[0], 1)

    q = q.reshape(-1, q.shape[-1])
    # print(q.shape)
    q = self.q_diff(q).reshape(*q_shape)
    # print(q.shape)
    out = solutions# if forcing else solutions[:, 0, :, :].unsqueeze(1)
    # print(out.shape)
    # print(q.shape)
    # print(out.shape)
    for b in range(q.shape[0]):
      out[b, :, scri_l[b], scrj_l[b]] += q[b]

    initial_hid = self.data_to_h0(eq_features)
    shape = out.shape
    #print(shape)
    out = self.encoder(seq_to_cnn(out.contiguous().unsqueeze(2)))
    #print(out.shape)
    x = self.encoder_1(out)
    #print(x.shape)
    x = x.view(shape[0], shape[1], x.shape[1], x.shape[2], x.shape[3])
    #print(x.shape)
    #print(initial_hid.shape)
    
    # if forcing:
    output, context = self.conv_lstm(x, x[:, 0, :, :, :], initial_hid)

    # else:
    #   context = (x[:, 0, :, :, :], initial_hid)
    #   output = [x]
    #   for t in range(solutions.shape[1]):
    #     # print(type(output[-1]))
    #     out, context = self.conv_lstm(output[-1], *context)
    #     context = (context[0][0], context[0][1])
    #     # print(type(context[0][0]))
    #     output.append(out[0])
    #   # print(output[-1].shape)
    #   output = torch.cat(output[1:], dim=1)
    
    if self.hidden_control is not None:
      h_diff = 0.#self.hidden_loss(output[0][:, :-1, :, :, :], x[:, 1:, :, :, :])
      #h_diff = output[0][:, :-1, :, :, :] - x[:, 1:, :, :, :]
    else:
      h_diff = 0.

    x = output[0] if forcing else output

    # print('here', x.shape)
    # print(x.shape)
    # print(seq_to_cnn(x).shape)
    x = self.decoder(seq_to_cnn(x))
    #print('feeee', torch.cat([x, out], dim=1).shape)
    #x = self.decoder_1(torch.cat([x, out], dim=1))
    #x = x + out
    x = self.decoder_1(x) 
    
    #print(x.shape)
    # print(x.shape)
    x = x.view(shape[0], solutions.shape[1], x.shape[1], x.shape[2], x.shape[3]).squeeze(2)
    #print(x.shape)
    return x, h_diff, context

