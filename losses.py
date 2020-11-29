import torch, torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import grad, Variable
import numpy as np

class InftyLoss(nn.Module):
  
  def forward(self, x, y):
    res = torch.abs(x - y)
    res = res.view(res.shape[0] * res.shape[1], -1)
    return torch.norm(res, p=np.Inf, dim=1).mean()


class WaveLoss(nn.Module):
  def __init__(self, order='l1', scale=1):
    super(WaveLoss, self).__init__()

    self.order = order
    self.scale =  scale

    if order == 'l1':
      self.loss = nn.L1Loss()
    elif order == 'l2':
      self.loss = nn.MSELoss()
    elif order == 'inf':
      self.loss = InftyLoss()

  def forward(self, x, y):
    return self.loss(x, y) * self.scale


def hessian_diag_u(inputs, net_forward):
  x = Variable(inputs, requires_grad=True).cuda()
  output = net_forward(x)
  # z = Variable(torch.zeros_like(output)).cuda()
  
  # print(net_forward(x).shape)
  # jacob = grad(output, x, grad_outputs=z, create_graph=True)[0]
  jacob = grad(output.sum(), x, create_graph=True, retain_graph=True)[0]
  # y = []
  # for g in jacob:
  #   for i, tmp in enumerate(g):
  #     print(tmp)
  #     g2 = grad(tmp, x[:, i], retain_graph=True, allow_unused=True)[0]
  #   y.append(g2)
  
  y = grad(jacob.sum(), x, create_graph=True)[0]
  # print(y)
  return y#torch.cat(y)

def hvp(f, x, v):
  return torch.autograd.functional.jvp(f, (x,), (v,))[1]



# class PhysicalLoss(nn.Module):
#   def __init__(self, order='l2', lam=1):
#     super(PhysicalLoss, self).__init__()
#     self.lam = 1
#     if order == 'l1':
#       self.loss = nn.L1Loss()
#     elif order == 'l2':
#       self.loss = nn.MSELoss()
#     elif order == 'inf':
#       self.loss = InftyLoss()

#   def forward(self, x, y inputs, net_forward): #x - our prediction, y - true answer, inputs - input of net
#     L_classic = self.loss(x, y)
#     # print(torch.autograd.functional.jacobian(net_forward, inputs))
#     #f = lambda x: torch.autograd.functional.jacobian(net_forward, x, create_graph=True)
#     #eps = 1e-6 
#     # print(f(inputs + eps))
#     #print(f(inputs))
#     #print((f(inputs + eps) - f(inputs)) / eps)
#     # print(torch.autograd.functional.jacobian(f, inputs))
#     #hessian_pred = torch.autograd.functional.hessian(lambda x: net_forward(x).sum(), inputs)
#     # hessian_pred = torch.autograd.functional.hvp(net_forward, inputs, torch.ones_like(inputs))
#     #fooo = hvp(f, inputs, torch.ones_like(inputs))
#     #print(fooo)
#     return hessian_pred# L_phys = lam * self.loss()




