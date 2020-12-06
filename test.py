from train_utils import get_batch
from utils import  Simulation, get_vp, ParamsGenerator, correlation_, RMSE_
from utils import RMSE_batch_one_picture, correlation_batch_one_picture, load_model
import numpy as np
import torch

class Test(object):
  def __init__(self, path_to_model, optimizer, model, device):
    self.loss_hist = {'train': [],
             'val': []}

    self.metrix_coeff = {'correlation': [],
                'RMSE': []}
    self.model = model
    self.optimizer = optimizer
    self.loss_hist, self.metrix_coeff, self.epoch = load_model(path_to_model, self.model, self.optimizer, 
                                                               self.loss_hist, self.metrix_coeff)
    self.device = device
    


  def test_videos(self, N_min=4, N_max=6, nx=80, nz=80, nt=50, batch_size=2):
    self.model.train(False)
    metrix_coeff = {'correlation': [],
                    'RMSE': []}
    predictions = []
    batch = get_batch(N_min, N_max, nx, nz, nt, batch_size)
    Input_u = batch['u_vp'][:, 20, 0, :, :]
    Input_vp = batch['u_vp'][:, 0, 1, :, :]
    Input_u /= 4.0e-09 
    Input_u = torch.from_numpy(Input_u).float()
    Input_vp = torch.from_numpy(Input_vp).float()
    Input_u = Input_u.unsqueeze(1).unsqueeze(1)
    Input_vp = Input_vp.unsqueeze(1)
    Labels_u = torch.from_numpy(batch['u_labels'][:, 20:, :, :]).float()
    Labels_u /= 4.0e-09 

    for i in range(0, nt-20-1):

      if self.model.model_type == 'AE':
        prediction_u = self.model(Input_u_vp[:, 0, :, :].to(self.device))
      elif self.model.model_type == 'GRU':
        # print(Input_u.shape, Input_vp.shape, Labels_u.shape)
        prediction_u, reg_loss = self.model(Input_u[:, :, 0, :, :].to(self.device),
                                            Input_vp.to(self.device))
        
      predictions.append(prediction_u)
      metrix_coeff['correlation'].append(correlation_batch_one_picture(prediction_u.to(self.device), Labels_u[:, i, :, :].to(self.device)).item())
      metrix_coeff['RMSE'].append(RMSE_batch_one_picture(prediction_u.to(self.device), Labels_u[:, i, :, :].to(self.device)).item())
      Input_u = prediction_u.unsqueeze(2)
      # Input_u *= 4.0e-09 
      for b in range(batch_size):
        Input_u[b, 0, 0, batch['scri_l'][b], batch['scrj_l'][b]] += batch['q'][b, i] * (batch['dT'][b] / batch['dd'][b])**2 / 4.0e-09
      
      # Input_u /= 4.0e-09

    
    return metrix_coeff, predictions, Labels_u, batch







