import torch
from utils import  Simulation, get_vp, ParamsGenerator, correlation_batch, RMSE_batch, correlation_batch_obo, RMSE_batch_obo
from train_utils import get_batch
from utils import correlation_batch_one_picture, RMSE_batch_one_picture
from models import seq_to_cnn
import time
import numpy as np
from tqdm.notebook import tqdm
from plottings import plot_valid, plot_metrics, plot_test
import gc

def test_model_model(model_1, model_2, N_min=4, N_max=6, nx=80, nz=80, nt=50, batch_size=2, device='cuda', separate_time=10):

  model_1.train(False)
  model_2.train(False)
  pred_list = []
  # print(nx, nz, nt)
  batch = get_batch(N_min, N_max, nx, nz, nt, batch_size)
  metrix_coeff = {'correlation': [], 'RMSE': []}

  # u = torch.from_numpy(batch['u_vp'][:, 10, 0, :, :]).float().unsqueeze(1)
  u = torch.from_numpy(batch['u'][:, 0, :, :]).float().unsqueeze(1) / 4.0e-09
  
  #u = torch.zeros_like(u)
  vp = torch.from_numpy(batch['vp']).float().unsqueeze(1)
  u_next = torch.from_numpy(batch['u_labels']) / 4.0e-09

  predictions = torch.zeros_like(u_next)

  for t in range(0, separate_time):
    # print(t)
    for b in range(batch_size):
      u[b, :, batch['scri_l'][b], batch['scrj_l'][b]] += batch['q'][b, t] * (batch['dT'][b] / batch['dd'][b])**2 / 4.0e-09
    
    # print(u.shape)
    if model_1.model_type == 'GRU':
      pred, _ = model_1((u).to(device), vp.to(device))
    elif model_1.model_type == 'AE':
      pred = model_1((u).to(device))
    
    if model_2.model_type == 'GRU':
      _, _ = model_2((u).to(device), vp.to(device))

    u = pred

    metrix_coeff['correlation'].append(correlation_batch_one_picture(pred.to(device), u_next[:, t, :, :].to(device)).item())
    metrix_coeff['RMSE'].append(RMSE_batch_one_picture(pred.to(device), u_next[:, t, :, :].to(device)).item())
    # pred_list.append(pred)
    predictions[:, t, :, :] = pred.squeeze(1)
    # print(predictions[:, t, :, :].shape, pred.shape)

    #print((predictions[:, t, :, :] == u.squeeze(1)).sum())
    

  for t in range(separate_time, nt - 1):
    # print(t)
    for b in range(batch_size):
      u[b, :, batch['scri_l'][b], batch['scrj_l'][b]] += batch['q'][b, t] * (batch['dT'][b] / batch['dd'][b])**2 / 4.0e-09
    
    # print(u.shape)
    if model_2.model_type == 'GRU':
      pred, _ = model_2((u).to(device), vp.to(device))
    elif model_2.model_type == 'AE':
      pred = model_2((u).to(device))
    
    u = pred

    metrix_coeff['correlation'].append(correlation_batch_one_picture(pred.to(device), u_next[:, t, :, :].to(device)).item())
    metrix_coeff['RMSE'].append(RMSE_batch_one_picture(pred.to(device), u_next[:, t, :, :].to(device)).item())
    # pred_list.append(pred)
    predictions[:, t, :, :] = pred.squeeze(1)
    # print(predictions[:, t, :, :].shape, pred.shape)

    #print((predictions[:, t, :, :] == u.squeeze(1)).sum())
    

  return predictions, u_next, vp, batch['f0_list']




# def test_diff_model(model, N_min=4, N_max=6, nx=80, nz=80, nt=50, batch_size=2, device='cuda', separate_time=10):

#   model.train(False)
#   pred_list = []
#   # print(nx, nz, nt)
#   batch = get_batch(N_min, N_max, nx, nz, nt, batch_size)
#   metrix_coeff = {'correlation': [], 'RMSE': []}

#   # u = torch.from_numpy(batch['u_vp'][:, 10, 0, :, :]).float().unsqueeze(1)
#   u = torch.from_numpy(batch['u'][:, separate_time, :, :]).float().unsqueeze(1) / 4.0e-09
  
#   #u = torch.zeros_like(u)
#   vp = torch.from_numpy(batch['vp']).float().unsqueeze(1)
#   u_next = torch.from_numpy(batch['u_labels']) / 4.0e-09

#   predictions = torch.zeros_like(u_next).float()

#   x = torch.zeros_like(u_next).float()
#   x[:, :separate_time, :, :] = torch.from_numpy(batch['u'][:, :separate_time, :, :]).float()

#   predictions[:, :separate_time, :, :] = u_next[:, :separate_time, :, :]
    
#   hidden = vp
#   context = None
#   # print(predictions[:, :separate_time, :, :].unsqueeze(2).shape)
#   # print(hidden.shape)

#   #pred, _, context = model((predictions[:, :separate_time, :, :]).to(device), hidden.to(device), context)

#   # for b in range(batch_size):
#   #     u[b, :, batch['scri_l'][b], batch['scrj_l'][b]] += batch['q'][b, t] * (batch['dT'][b] / batch['dd'][b])**2 / 4.0e-09

#   #pred, _, context = model((u).to(device), hidden.to(device), context)

#   # print(pred[:, -2, :, :].unsqueeze(1).shape)
#   # u = pred[:, -2, :, :].unsqueeze(1)
#   for t in range(separate_time, nt - 1):
#     # print(t)
#     #for b in range(batch_size):
#     #  u[b, :, batch['scri_l'][b], batch['scrj_l'][b]] += batch['q'][b, t] * (batch['dT'][b] / batch['dd'][b])**2 / 4.0e-09
    
#     # print(u.shape)
#     if model.model_type == 'GRU':
#       pred, _ = model((u).to(device), hidden.to(device))
#     elif model.model_type == 'LSTM':
#       pred, _, context = model(x, hidden.to(device), context)
#     elif model.model_type == 'AE':
#       pred = model((u).to(device))
    
#     predictions[:, t, :, :] =  pred[:, t, :, :]
#     x[:, t, :, :] =  pred[:, t, :, :]
#     #u = pred
#     # hidden = pred[:, -1, :, :]#u
#     #metrix_coeff['correlation'].append(correlation_batch_one_picture(pred.to(device), u_next[:, t, :, :].to(device)).item())
#     #metrix_coeff['RMSE'].append(RMSE_batch_one_picture(pred.to(device), u_next[:, t, :, :].to(device)).item())
#     # pred_list.append(pred)
#     #predictions[:, t, :, :] = pred.squeeze(1)
#     # print(predictions[:, t, :, :].shape, pred.shape)

#     #print((predictions[:, t, :, :] == u.squeeze(1)).sum())
    

#   return predictions, u_next, vp, batch['f0_list']
def test_diff_model(model, N_min=4, N_max=6, nx=80, nz=80, nt=50, batch_size=2, device='cuda', separate_time=10):

  model.train(False)
  pred_list = []
  batch = get_batch(N_min, N_max, nx, nz, nt, batch_size)
  metrix_coeff = {'correlation': [], 'RMSE': []}

  u = torch.from_numpy(batch['u'][:, separate_time, :, :]).float().unsqueeze(1).to(device)
  
  vp = torch.from_numpy(batch['vp']).float().unsqueeze(1)
  u_next = torch.from_numpy(batch['u_labels']) #/ 4.0e-09

  predictions = torch.zeros_like(u_next).float()

  predictions[:, :separate_time, :, :] = u_next[:, :separate_time, :, :]




  hidden = vp.to(device)
  context = None

  for t in range(0, nt - 1):

    if model.model_type == 'GRU':
      pred, _ = model((u).to(device), hidden.to(device))
    elif model.model_type == 'LSTM':
      if t==0:
        pred, _, context = model(u, torch.zeros_like(hidden), hidden)
        
      else:

        shape = u.shape
 
        out = model.encoder(seq_to_cnn(u.contiguous().unsqueeze(2)))
   
        x = model.encoder_1(out)
   
        x = x.view(shape[0], shape[1], x.shape[1], x.shape[2], x.shape[3])

        output, context = model.conv_lstm(x, context[0][0], context[0][-1])
        
        if model.hidden_control is not None:
          h_diff = 0.#self.hidden_loss(output[0][:, :-1, :, :, :], x[:, 1:, :, :, :])
          #h_diff = output[0][:, :-1, :, :, :] - x[:, 1:, :, :, :]
        else:
          h_diff = 0.

        x = output[0]

        x = model.decoder(seq_to_cnn(x))

        x = model.decoder_1(x) 

        pred = x.view(shape[0], shape[1], x.shape[1], x.shape[2], x.shape[3]).squeeze(2)

    elif model.model_type == 'AE':
      pred = model((u).to(device))

    # pred_list.append(pred)
    u = torch.from_numpy(batch['u'][:, t, :, :]).float().unsqueeze(1).to(device)

    

  return predictions, u_next, vp, batch['f0_list']







def test_diff_model__(model, N_min=4, N_max=6, nx=80, nz=80, nt=50, batch_size=2, device='cuda', separate_time=0, mode='itself'):

  model.train(False)
  # pred_list = []
  # print(nx, nz, nt)
  batch = get_batch(N_min, N_max, nx, nz, nt, batch_size)
  
  factor = 1.#4.0e-09
  #factor = batch['u'].std()
  # factor_vp = batch['dt']

  # X = torch.from_numpy(batch['u_vp']).float()
  # X[:, :, 0, :, :] /= factor

  X = torch.from_numpy(batch['u']).float()
  X /= factor
  X_vp = torch.from_numpy(batch['vp']).float()

  X_labels = torch.from_numpy(batch['u_labels']).float()[:, 1:, :, :]
  X_labels /= factor 
  q_prev_curr = torch.stack([torch.Tensor(batch['q'][:, : -2]), torch.Tensor(batch['q'][:, 1: -1])], dim=-1)
    

  predictions = torch.zeros(size=(X_labels.shape[0], nt, X_labels.shape[2], X_labels.shape[3])).float()
  predictions[:, 1:separate_time + 1, :, :] = X_labels[:, :separate_time, :, :]


  if model.model_type == 'AE':
    start = time.time()
    # print(X[:, :-1, :, :].shape)
    # predictions = model(X[:, :-1, 0, :, :].to(device))
    predictions = model(X[:, : -1, :, :].to(device))
    t_nn += (time.time() - start) / batch_size
    reg_loss = 0.
  else:
    start = time.time()
    # predictions, reg_loss = model(X[:, :-1, 0, :, :].to(device), X[:, 0, 1, :, :].unsqueeze(1).to(device))
    predictions, reg_loss, _ = model(X[:, 1: -1, :, :].to(device), X_vp.unsqueeze(1).to(device), q_prev_curr.to(device), batch['scri_l'], batch['scrj_l'], forcing=None)
    for b in range(predictions.shape[0]):
      # add = torch.Tensor(batch['q'][b] * batch['dt'][b]**2 / batch['dd'][b]**2)
      add = torch.Tensor(batch['q'][b])
      #print(add.shape)
      predictions[b, :, batch['scri_l'][b], batch['scrj_l'][b]] += add[1: -1].to(device)
    t_nn += (time.time() - start) / batch_size

    fact = 1.#predictions[:, :t + separate_time, :, :].std() / pred[:, -1, :, :].std() #* np.sqrt(2.5)

    predictions[:, t + separate_time + 1, :, :] = pred[:, -1, :, :] * fact
    
    if mode == 'itself':
      X = predictions[:, :t + separate_time + 1 + 1, :, :]
    elif mode == 'help_diff':
      X = X_labels[:, :t + separate_time + 1, :, :]
    # add_source(X, batch)


    

  return predictions[:, 1:, :, :], X_labels, X_vp.unsqueeze(1), batch['f0_list']

def add_source(X, batch):
  tm = X.shape[1]
  for b in range(X.shape[0]):
    X[b, :, batch['scri_l'][b], batch['scrj_l'][b]] += torch.from_numpy(batch['q'][b, :tm] / batch['dd'][b] ** 2 * (batch['dT'][b]) ** 2).float()
    





def testing(model, optimizer, loss_hist, n_validation_batches, 
            device, N_min, N_max, nx, nz, nt, loss, batch_size, metrix_coeff, forcing=True, gif=False):
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
    
    factor = 1.#4.0e-09
    #factor = batch['u'].std()
    # factor_vp = batch['dt']

    # X = torch.from_numpy(batch['u_vp']).float()
    # X[:, :, 0, :, :] /= factor

    X = torch.from_numpy(batch['u']).float()
    X /= factor
    X_vp = torch.from_numpy(batch['vp']).float()

    X_labels = torch.from_numpy(batch['u_labels']).float()[:, 1:, :, :]
    X_labels /= factor 
    q_prev_curr = torch.stack([torch.Tensor(batch['q'][:, : -2]), torch.Tensor(batch['q'][:, 1: -1])], dim=-1)
    # X[:, :, 1, :, :] *= factor_vp.reshape(-1, 1, 1, 1)
    t_fd = 0
    if model.model_type == 'AE':
      t_nn = 0
      start = time.time()
      # predictions = model(X[:, :-1, 0, :, :].to(device))
      predictions = model(X[:, :-1, :, :].to(device))
      reg_loss = 0.
      t_nn += (time.time() - start) / batch_size
    else:
      t_nn = 0
      start = time.time()
      # predictions, reg_loss = model(X[:, :-1, 0, :, :].to(device), X[:, 0, 1, :, :].unsqueeze(1).to(device))
      predictions, reg_loss, _ = model(X[:, 1: -1, :, :].to(device), X_vp.unsqueeze(1).to(device), q_prev_curr.to(device), batch['scri_l'], batch['scrj_l'], forcing=forcing)
      t_nn += (time.time() - start) / batch_size * (1 - 0.65)
      for b in range(predictions.shape[0]):
        # add = torch.Tensor(batch['q'][b] * batch['dt'][b]**2 / batch['dd'][b]**2)
        add = torch.Tensor(batch['q'][b])
        #print(add.shape)
        predictions[b, :, batch['scri_l'][b], batch['scrj_l'][b]] += add[1: -1].to(device)
      

      
    t_fd += batch['exec_time'][0]
    
    print('time nn: ', t_nn,'time dm: ',  t_fd)

    # loss_t = loss(predictions, X[:, 1: , 0, :, :].to(device)) 
    # print(predictions.shape, X_labels.shape)
    loss_t = loss(predictions, X_labels.to(device)) 
    loss_t += reg_loss

    val_loss += loss_t.item()
    if number_i == plot_this:
      # plot_valid(X_labels, predictions, X[:, 0, -1, :, :], batch_size, batch['f0_list'])
      plot_test(X_labels, predictions, X_vp, batch_size, batch['f0_list'], gif=True)
    
    corr_list.append(correlation_batch(predictions.to(device), X_labels.to(device)).item())
    rmse_list.append(RMSE_batch(predictions.to(device), X_labels.to(device)).item())
  
  val_loss /= n_validation_batches
  loss_hist['val'].append(val_loss)

  
  metrix_coeff['correlation'].append(np.mean(corr_list))
  metrix_coeff['RMSE'].append(np.mean(rmse_list))
  
  del X
  gc.collect()
  torch.cuda.empty_cache()

def testing_from0(model, optimizer, loss_hist, n_validation_batches, 
            device, N_min, N_max, nx, nz, nt, loss, batch_size, metrix_coeff, start_nn_t=0, forcing=True):
  val_loss=0
  model.train(False)
  plot_this = np.random.randint(low=0, high=n_validation_batches)
  corr_list = []
  rmse_list = []

  times_nn = []
  
  batch = get_batch(N_min, N_max, nx, nz, nt, batch_size=1)
  optimizer.zero_grad()
  
  factor = 1.#4.0e-09


  X = torch.from_numpy(batch['u']).float()#[:, :start_nn_t, :, :]
  
  X /= factor
  X_vp = torch.from_numpy(batch['vp']).float()

  X_labels = torch.from_numpy(batch['u_labels']).float()[:, 1:, :, :]
  answers = torch.zeros_like(X_labels)
  X_labels /= factor 
  q_prev_curr = torch.stack([torch.Tensor(batch['q'][:, : -2]), torch.Tensor(batch['q'][:, 1: -1])], dim=-1)
  t_nn = 0

  X = X[:, :start_nn_t]
  q = q_prev_curr[:, :start_nn_t - 1]
  
  predictions, reg_loss, _ = model(X[:, 1: , :, :].to(device), X_vp.unsqueeze(1).to(device), q.to(device), batch['scri_l'], batch['scrj_l'], forcing=forcing)

  for b in range(predictions.shape[0]):

    add = torch.Tensor(batch['q'][b, :start_nn_t])

    predictions[b, :, batch['scri_l'][b], batch['scrj_l'][b]] += add[1: ].to(device)


  print('end with help',X.shape, predictions.shape)
  for i in range(start_nn_t, nt - 1):
    X_new = torch.zeros(X.shape[0], X.shape[1] + 1, X.shape[2], X.shape[3])
    X_new[:, :-1] = X
    
    X_new[:, -1, :, :] = predictions[:, -1, :, :] 
    # print(q_prev_curr.shape)
    q = q_prev_curr[:, :i]
    print(X.shape, q.shape)
    X = X_new
    # print('X ',X.shape)
    predictions, reg_loss, _ = model(X[:, 1: , :, :].to(device), X_vp.unsqueeze(1).to(device), q.to(device), batch['scri_l'], batch['scrj_l'], forcing=forcing)
    # print('Pr ',predictions.shape)
    for b in range(predictions.shape[0]):
      add = torch.Tensor(batch['q'][b, :i])
      predictions[b, :, batch['scri_l'][b], batch['scrj_l'][b]] += add[:].to(device)

  plot_test(X_labels, predictions, X_vp, batch_size, batch['f0_list'])
  
  # corr_list.append(correlation_batch(predictions.to(device), X_labels.to(device)).item())
  # rmse_list.append(RMSE_batch(predictions.to(device), X_labels.to(device)).item())
  

  
  # metrix_coeff['correlation'].append(np.mean(corr_list))
  # metrix_coeff['RMSE'].append(np.mean(rmse_list))
  
  del X
  gc.collect()
  torch.cuda.empty_cache()


