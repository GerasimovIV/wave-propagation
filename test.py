from train_utils import get_batch
from utils import  Simulation, get_vp, ParamsGenerator, correlation_, RMSE_
import numpy as np
import torch

class Test(object):
  def __init__(self, path_to_model, model, optimizer):
    self.loss_hist = {'train': [],
             'val': []}

    self.metrix_coeff = {'correlation': [],
                'RMSE': []}
    self.model = model
    self.optimizer = optimizer
    self.loss_hist, self.metrix_coeff, self.epoch = load_model(path_to_model, self.model, self.optimizer, 
                                                               self.loss_hist, self.metrix_coeff)

    




  def test_one_video(self, N_min=4, N_max=6, nx=80, nz=80, nt=50):
    self.model.train(False)
    metrix_coeff = {}

    if self.model.model_type == 'AE':
      video = get_test_video(N_min, N_max, nx, nz, nt)

    X = make_input_data(video['u'][0], video['dd'], video['dt'], 
                        video['srci'], video['srcj'], video['q'][0], d['vp'])
  
    optimizer.zero_grad()
    time_for_predictions = 0
    predictions_list = []
    if model.model_type == 'AE':
      for i in range(1, X.shape[1]):
        start = time.time()
        prediction = model(X[0, 0, 0, :, :].to(device))
        t_nn += (time.time() - start)
        predictions_list.append(prediction[0,0,:,:])
        X = make_input_data(prediction[0, 0, :, :], video['dd'], video['dt'], 
                        video['srci'], video['srcj'], video['q'][i], d['vp'])

    predictions = torch.tensor(predictions_list)
    metrix_coeff['correlation'].append(correlation_(predictions.to(device), video['u'][1:, :, :].to(device)).item())
    metrix_coeff['RMSE'].append(RMSE_(predictions.to(device), video['u'][1:, :, :].to(device)).item())


def get_test_video(N_min=4, N_max=6, nx=80, nz=80, nt=50):
  batch_sols = []
  initial_features = []
  exec_time = []
  results_data = {}
    
  params_generator = ParamsGenerator(N_min, N_max, nx, nz, nt)
  Simulator, t_i = params_generator.set_simulator_params()

  exec_time = t_i
  

  results_data['u'] = np.array(Simulator.field2d.transpose((2, 0, 1)))
  results_data['vp'] = np.array(Simulator.vp)
  results_data['q'] = Simulator.wav
  results_data['exec_time'] = exec_time
  results_data['dd'] = Simulator.dd
  results_data['dt'] = Simulator.dt
  results_data['f0'] = params_generator.f0
  results_data['srci'] = Simulator.srci
  results_data['srcj'] = Simulator.srcj

  return results_data



def make_input_data(u, dd, dt, srci, srcj, q, vp):
  u[srci, srcj] += q / dd ** 2 * dt ** 2
  u = np.expand_dims(u,axis=0)
  X = np.expand_dims(np.stack((u, np.repeat(np.expand_dims(vp, axis=0), u.shape[0], axis=0)), axis=1), axis=0)
  # d['u'][:, d['srci'], d['srcj']] += d['q'] / d['dd'] ** 2 * d['dt'] ** 2
  # X = np.expand_dims(np.stack((d['u'], np.repeat(np.expand_dims(d['vp'], axis=0), d['u'].shape[0], axis=0)), axis=1), axis=0)
  # # print(X.shape)
  factor = X[:, :-1, 0, :, :].std()
  X = torch.from_numpy(X).float()
  X[:, :, 0, :, :] /= factor
  return X





