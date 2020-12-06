from utils import  Simulation, get_vp, ParamsGenerator, correlation_batch, RMSE_batch, correlation_batch_obo, RMSE_batch_obo
import numpy as np
from plottings import plot_valid, plot_metrics
from tqdm.notebook import tqdm 
import torch
import time
import gc
from scipy.stats import pearsonr

def get_batch(N_min, N_max, nx, nz, nt, batch_size=20):
  
  # t_init = np.random.randint(low=0, high=400)
  t_init = 0
  batch_sols = []
  initial_features = []
  exec_time = []
  dd = []
  dt = []
  dT = []
  q = []
  f0_list = []
  srci_l = []
  srcj_l = []
  results_data = {}
  u_labels = []
  for b in range(batch_size):
    
    params_generator = ParamsGenerator(N_min, N_max, nx, nz, nt + t_init)
    Simulator, t_i = params_generator.set_simulator_params()
    exec_time.append(t_i)
    q.append(Simulator.wav[::params_generator.coeff_rarefaction][t_init:])
    u_labels.append(np.copy(np.array(Simulator.field2d.transpose((2, 0, 1)))[::params_generator.coeff_rarefaction][t_init:]))
    Simulator.field2d[Simulator.srci, Simulator.srcj] += Simulator.wav / Simulator.dd ** 2 * (Simulator.dt) ** 2 * params_generator.coeff_rarefaction**2
    batch_sols.append(np.array(Simulator.field2d.transpose((2, 0, 1)))[::params_generator.coeff_rarefaction][t_init:])
    initial_features.append(np.array(Simulator.vp))
    dd.append(Simulator.dd)
    dt.append(Simulator.dt)
    dT.append(Simulator.dt*params_generator.coeff_rarefaction)
    f0_list.append(params_generator.f0)
    
    srci_l.append(Simulator.srci)
    srcj_l.append(Simulator.srcj)
  # print(np.stack(batch_sols).shape)
  # results_data['u_vp'] = np.stack((np.stack(batch_sols), np.repeat(np.expand_dims(np.stack(initial_features), axis=1), batch_sols[0].shape[0], axis=1)), axis=2)
  results_data['vp'] = np.stack(initial_features)
  results_data['u'] = np.stack(batch_sols)
  results_data['q'] = np.stack(q)
  results_data['exec_time'] = exec_time
  results_data['dd'] = np.stack(dd)
  results_data['dt'] = np.stack(dt)
  results_data['f0_list'] = np.stack(f0_list)
  results_data['u_labels'] = np.stack(u_labels)[:, 1:, :, :]
  results_data['scri_l'] = np.stack(srci_l)
  results_data['scrj_l'] = np.stack(srcj_l)
  results_data['dT'] = np.stack(dT)

  return results_data
  # return np.stack((np.stack(batch_sols), np.repeat(np.expand_dims(np.stack(initial_features), axis=1), nt, axis=1)), axis=2), t / batch_size

def get_video_OBO(N_min, N_max, nx, nz, nt):
  pass
  # results_data = {}

  # t_init = np.random.randint(low=0, high=400)
  # params_generator = ParamsGenerator(N_min, N_max, nx, nz, nt + t_init)
  # Simulator, t_i = params_generator.set_simulator_params()

  # results_data['vp'] = Simulator.vp
  # results_data['u'] = np.array(Simulator.field2d.transpose((2, 0, 1)))[t_init: ]
  # results_data['q'] = Simulator.wav[t_init: ]
  # results_data['dd'] = Simulator.dd
  # results_data['dt'] = Simulator.dt
  # results_data['f0'] = params_generator.f0
  # results_data['srci'] = Simulator.srci
  # results_data['srcj'] = Simulator.srcj
  # results_data['t_init'] = t_init

  # return results_data

def get_inp_OBO_from_video(data, tn, batch_size=60):# result_data from get_video_OBO
  t_init = data['t_init']
  # ti = tn // (data['u'].shape[1] * data['u'].shape[1])
  # xi = (tn - ti * (data['u'].shape[1] * data['u'].shape[1])) % data['u'].shape[1]
  # zi = (tn - ti * (data['u'].shape[1] * data['u'].shape[1])) // data['u'].shape[2]
  inp = np.zeros((batch_size, 10)) # x, z, t, vp, u, srci, srcj, q, dd, dt
  for j in range(batch_size):
    ti = tn // (data['u'].shape[1] * data['u'].shape[1])
    xi = (tn - ti * (data['u'].shape[1] * data['u'].shape[1])) % data['u'].shape[1]
    zi = (tn - ti * (data['u'].shape[1] * data['u'].shape[1])) // data['u'].shape[2]
    # print(xi, zi, ti, tn)
    inp[j, :] = np.array([xi*data['dd'], zi*data['dd'], (ti + t_init) * data['dt'],
                            data['vp'][xi, zi], data['u'][ti, xi, zi], data['srci']*data['dd'],
                            data['srcj']*data['dd'], data['q'][ti], data['dd'], data['dt']])
    tn+=1
  
  return inp

def get_label_OBO_from_video(data, tn, batch_size=60):# result_data from get_video_OBO
  t_init = data['t_init']
  tn += data['u'].shape[-1] * data['u'].shape[-2]
  # ti = tn // (data['u'].shape[1] * data['u'].shape[1])
  # xi = (tn - ti * (data['u'].shape[1] * data['u'].shape[1])) % data['u'].shape[1]
  # zi = (tn - ti * (data['u'].shape[1] * data['u'].shape[1])) // data['u'].shape[2]
  inp = np.zeros((batch_size, 10)) # x, z, t, vp, u, srci, srcj, q, dd, dt
  for j in range(batch_size):
    ti = tn // (data['u'].shape[1] * data['u'].shape[1])
    xi = (tn - ti * (data['u'].shape[1] * data['u'].shape[1])) % data['u'].shape[1]
    zi = (tn - ti * (data['u'].shape[1] * data['u'].shape[1])) // data['u'].shape[2]
    # print(xi, zi, ti, tn)
    inp[j, :] = np.array([xi*data['dd'], zi*data['dd'], (ti + t_init) * data['dt'],
                            data['vp'][xi, zi], data['u'][ti, xi, zi], data['srci']*data['dd'],
                            data['srcj']*data['dd'], data['q'][ti], data['dd'], data['dt']])
    tn+=1
  
  return inp




def train(model, optimizer, loss_hist, epoch_time_nn, N_min, N_max, nx, nz, nt,
          epoch_time_fd, n_batches_per_epoch, batch_size, device, loss, batch_factors):#, scheduler):
  train_loss = 0
  model.train(True)
  t_nn = 0.
  t_fd = 0.
  #batch_factors = []

  for _ in tqdm(range(n_batches_per_epoch)):
    # if model.model_type == 'AE':
    #   batch = get_batch(N_min, N_max, nx, nz, nt, batch_size)
    # else:
    #   #raise NotImplementedError('RNN to be continued')
    #   batch = get_batch(N_min, N_max, nx, nz, nt, batch_size)
    batch = get_batch(N_min, N_max, nx, nz, nt, batch_size)
    optimizer.zero_grad()
    
    factor = 4.0e-09 
    #factor = batch['u'].std()
    #batch_factors.append(factor)
    # factor_vp = batch['dt']

    # X = torch.from_numpy(batch['u_vp']).float()
    # X[:, :, 0, :, :] /= factor
    X = torch.from_numpy(batch['u']).float()
    X /= factor
    X_vp = torch.from_numpy(batch['vp']).float()

    X_labels = torch.from_numpy(batch['u_labels']).float()
    X_labels /= factor

    # print(factor)
    # X[:, :, 1, :, :] /= factor_vp.reshape(-1, 1, 1, 1)

    if model.model_type == 'AE':
      start = time.time()
      # print(X[:, :-1, :, :].shape)
      # predictions = model(X[:, :-1, 0, :, :].to(device))
      predictions = model(X[:, :-1, :, :].to(device))
      t_nn += (time.time() - start) / batch_size
      reg_loss = 0.
    else:
      start = time.time()
      # predictions, reg_loss = model(X[:, :-1, 0, :, :].to(device), X[:, 0, 1, :, :].unsqueeze(1).to(device))
      predictions, reg_loss, _ = model(X[:, :-1, :, :].to(device), X_vp.unsqueeze(1).to(device))
      t_nn += (time.time() - start) / batch_size

      
    # t_fd += batch['exec_time']
    
    # print(t_nn, t_fd)

    # loss_t = loss(predictions, X[:, 1: , 0, :, :].to(device)) 
    # print(predictions.shape, X_labels.shape)
    loss_t = loss(predictions, X_labels.to(device)) 
    loss_t += reg_loss

    loss_t.backward()
    optimizer.step()
    # scheduler.step()
    
    train_loss += loss_t.item()
    
    del X, predictions
    gc.collect()
    torch.cuda.empty_cache()
      
  epoch_time_nn.append(t_nn / n_batches_per_epoch)
  epoch_time_fd.append(t_fd / n_batches_per_epoch)

  train_loss /= n_batches_per_epoch
  
  loss_hist['train'].append(train_loss)


def validate(model, optimizer, loss_hist, n_validation_batches, 
            device, N_min, N_max, nx, nz, nt, loss, batch_size, metrix_coeff):
  val_loss=0
  model.train(False)
  plot_this = np.random.randint(low=0, high=n_validation_batches)
  corr_list = []
  rmse_list = []

  times_nn = []

  for number_i in tqdm(range(n_validation_batches)):
    
    # if model.model_type == 'AE':
    #   batch = get_batch(N_min, N_max, nx, nz, nt, batch_size)
    # else:
    #   batch = get_batch(N_min, N_max, nx, nz, nt, batch_size)
      #raise NotImplementedError('RNN to be continued')
    batch = get_batch(N_min, N_max, nx, nz, nt, batch_size)
    optimizer.zero_grad()
    
    factor = 4.0e-09 
    #factor = batch['u'].std()
    # factor_vp = batch['dt']

    # X = torch.from_numpy(batch['u_vp']).float()
    # X[:, :, 0, :, :] /= factor

    X = torch.from_numpy(batch['u']).float()
    X /= factor
    X_vp = torch.from_numpy(batch['vp']).float()

    X_labels = torch.from_numpy(batch['u_labels']).float()
    X_labels /= factor 
    # X[:, :, 1, :, :] *= factor_vp.reshape(-1, 1, 1, 1)
    t_nn = 0
    if model.model_type == 'AE':
      start = time.time()
      # predictions = model(X[:, :-1, 0, :, :].to(device))
      predictions = model(X[:, :-1, :, :].to(device))
      reg_loss = 0.
      t_nn += (time.time() - start) / batch_size
    else:
      start = time.time()
      # predictions, reg_loss = model(X[:, :-1, 0, :, :].to(device), X[:, 0, 1, :, :].unsqueeze(1).to(device))
      predictions, reg_loss, _ = model(X[:, :-1, :, :].to(device), X_vp.unsqueeze(1).to(device))
      t_nn += time.time() - start

    times_nn.append(t_nn / batch_size)
    #predictions = model(X[:, :-1, 0, :, :].to(device))
    # predictions = model(X[:, :-1, 0, :, :].to(device), X[:, 0, 0, :, :].unsqueeze(1).to(device))
    # loss_t = loss(predictions, X[:, 1: , 0, :, :].to(device))  
    # print(predictions.shape, X_labels.shape)
    loss_t = loss(predictions, X_labels.to(device)) 
    loss_t += reg_loss

    val_loss += loss_t.item()
    if number_i == plot_this:
      # plot_valid(X_labels, predictions, X[:, 0, -1, :, :], batch_size, batch['f0_list'])
      plot_valid(X_labels, predictions, X_vp, batch_size, batch['f0_list'])
    
    corr_list.append(correlation_batch(predictions.to(device), X_labels.to(device)).item())
    rmse_list.append(RMSE_batch(predictions.to(device), X_labels.to(device)).item())
  
  val_loss /= n_validation_batches
  loss_hist['val'].append(val_loss)

  
  metrix_coeff['correlation'].append(np.mean(corr_list))
  metrix_coeff['RMSE'].append(np.mean(rmse_list))
  
  del X, predictions
  gc.collect()
  torch.cuda.empty_cache()
  



def train_OBO(model, optimizer, loss_hist, N_min, N_max, nx, nz, nt,
              n_videos_per_epoch, batch_size, device, loss):
  train_loss = 0
  model.train(True)

  for _ in tqdm(range(n_videos_per_epoch)):
    video = get_video_OBO(N_min=N_min, N_max=N_max, nx=nx, nz=nz, nt=nt)
    for tn in tqdm(range(0, video['u'].shape[-1]*video['u'].shape[-2]*(video['u'].shape[-3] - 1), batch_size)):
      inputs = get_inp_OBO_from_video(video, tn, batch_size=batch_size)
      labels = get_label_OBO_from_video(video, tn, batch_size=batch_size)[:, 4]
      #x, z, t, vp, u, si, sj, wav
      optimizer.zero_grad()
    
      factor = inputs[:, 4].std() 

      inputs = torch.from_numpy(inputs).float()
      inputs[:, 4] /= factor
      
      labels = torch.from_numpy(labels).float()
      labels = labels.unsqueeze(1)
      predictions, residual = model(inputs.to(device))

      loss_t = loss(predictions, labels.to(device))
      loss_t += residual.mean()

      loss_t.backward()
      optimizer.step()
      # scheduler.step()
      
      train_loss += loss_t.item()
      
      del inputs, predictions
      gc.collect()
      torch.cuda.empty_cache()


  train_loss /= n_videos_per_epoch
  
  loss_hist['train'].append(train_loss)


def validate_obo(model, optimizer, loss_hist, metrix_coeff, N_min, N_max, nx, nz, nt,
              n_videos_per_epoch, batch_size, device, loss):
  val_loss = 0
  model.train(False)

  for _ in tqdm(range(n_videos_per_epoch)):
    video = get_video_OBO(N_min=N_min, N_max=N_max, nx=nx, nz=nz, nt=nt)
    for tn in tqdm(range(0, video['u'].shape[-1]*video['u'].shape[-2]*(video['u'].shape[-3] - 1), batch_size)):
      inputs = get_inp_OBO_from_video(video, tn, batch_size=batch_size)
      labels = get_label_OBO_from_video(video, tn, batch_size=batch_size)[:, 4]
      #x, z, t, vp, u, si, sj, wav
      optimizer.zero_grad()
    
      factor = inputs[:, 4].std() 

      inputs = torch.from_numpy(inputs).float()
      inputs[:, 4] /= factor
      
      labels = torch.from_numpy(labels).float()
      labels = labels.unsqueeze(1)
      predictions, residual = model(inputs.to(device))

      loss_t = loss(predictions, labels.to(device))
      loss_t += residual.mean()

      loss_t.backward()
      optimizer.step()
      # scheduler.step()
      
      val_loss += loss_t.item()
      
      # if number_i == plot_this:
      #   plot_valid_obo(inputs, predictions, batch_size)

      

  val_loss /= n_videos_per_epoch
  
  loss_hist['val'].append(val_loss)

  metrix_coeff['correlation'].append(correlation_batch_obo(predictions.to(device), labels.to(device)).item())
  metrix_coeff['RMSE'].append(RMSE_batch_obo(predictions.to(device), labels.to(device)).item())

