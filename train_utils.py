from utils import  Simulation, get_vp, ParamsGenerator
import numpy as np
from plottings import plot_valid, plot_metrics
from tqdm.notebook import tqdm 
import torch
import time
import gc

def get_batch_AE(N_min, N_max, nx, nz, nt, batch_size=20):
  
  t_init = np.random.randint(low=0, high=1000)
  batch_sols = []
  initial_features = []
  exec_time = []
  dd = []
  dt = []
  q = []
  results_data = {}

  for b in range(batch_size):
    
    params_generator = ParamsGenerator(N_min, N_max, nx, nz, nt + t_init)
    Simulator, t_i = params_generator.set_simulator_params()

    exec_time.append(t_i)
    q.append(Simulator.wav[t_init: ])
    batch_sols.append(np.array(Simulator.field2d.transpose((2, 0, 1)))[t_init: ])
    initial_features.append(np.array(Simulator.vp))
    dd.append(Simulator.dd)
    dt.append(Simulator.dt)

  results_data['u_vp'] = np.stack((np.stack(batch_sols), np.repeat(np.expand_dims(np.stack(initial_features), axis=1), nt, axis=1)), axis=2)
  results_data['q'] = np.stack(q)
  results_data['exec_time'] = np.mean(exec_time)
  results_data['dd'] = np.stack(dd)
  results_data['dt'] = np.stack(dt)

  return results_data
  # return np.stack((np.stack(batch_sols), np.repeat(np.expand_dims(np.stack(initial_features), axis=1), nt, axis=1)), axis=2), t / batch_size






def train(model, optimizer, loss_hist, epoch_time_nn, N_min, N_max, nx, nz, nt,
          epoch_time_fd, n_batches_per_epoch, batch_size, device, loss):
  train_loss = 0
  model.train(True)
  t_nn = 0.
  t_fd = 0.
  for _ in tqdm(range(n_batches_per_epoch)):
    if model.model_type == 'AE':
      batch = get_batch_AE(N_min, N_max, nx, nz, nt, batch_size)
    else:
      #raise NotImplementedError('RNN to be continued')
      batch = get_batch_AE(N_min, N_max, nx, nz, nt, batch_size)
    optimizer.zero_grad()
    
    factor = batch['u_vp'][:, :-1, 0, :, :].std()
    X = torch.from_numpy(batch['u_vp']).float()
    X[:, :, 0, :, :] /= factor

    if model.model_type == 'AE':
      start = time.time()
      predictions = model(X[:, :-1, 0, :, :].to(device))
      t_nn += (time.time() - start) / batch_size
    else:
      start = time.time()
      predictions = model(X[:, :-1, 0, :, :].to(device), X[:, 0, 0, :, :].unsqueeze(1).to(device))
      t_nn += (time.time() - start) / batch_size

    t_fd += batch['exec_time']
    
    # print(t_nn, t_fd)

    loss_t = loss(predictions, X[:, 1: , 0, :, :].to(device))

    loss_t.backward()
    optimizer.step()
    
    train_loss += loss_t.item()
    
    del X, predictions
    gc.collect()
    torch.cuda.empty_cache()
      
  epoch_time_nn.append(t_nn / n_batches_per_epoch)
  epoch_time_fd.append(t_fd / n_batches_per_epoch)

  train_loss /= n_batches_per_epoch
  
  loss_hist['train'].append(train_loss)


def validate(model, optimizer, loss_hist, n_validation_batches, 
            device, N_min, N_max, nx, nz, nt, loss, batch_size):
  val_loss=0
  model.train(False)
  for _ in tqdm(range(n_validation_batches)):
    
    if model.model_type == 'AE':
      batch = get_batch_AE(N_min, N_max, nx, nz, nt, batch_size)
    else:
      batch = get_batch_AE(N_min, N_max, nx, nz, nt, batch_size)
      #raise NotImplementedError('RNN to be continued')
    optimizer.zero_grad()
    
    factor = batch['u_vp'][:, :-1, 0, :, :].std()

    X = torch.from_numpy(batch['u_vp']).float()
    X[:, :, 0, :, :] /= factor

    #predictions = model(X[:, :-1, 0, :, :].to(device))
    predictions = model(X[:, :-1, 0, :, :].to(device), X[:, 0, 0, :, :].unsqueeze(1).to(device))
    loss_t = loss(predictions, X[:, 1: , 0, :, :].to(device))  
    
    val_loss += loss_t.item()

  val_loss /= n_validation_batches
  loss_hist['val'].append(val_loss)

  
  plot_valid(X, predictions, batch_size)

  del X, predictions
  gc.collect()
  torch.cuda.empty_cache()




def test():
  pass