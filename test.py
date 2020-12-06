import torch
from train_utils import get_batch
from utils import correlation_batch_one_picture, RMSE_batch_one_picture
from models import seq_to_cnn
import time

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
  # print(nx, nz, nt)
  batch = get_batch(N_min, N_max, nx, nz, nt, batch_size)
  metrix_coeff = {'correlation': [], 'RMSE': []}

  u = torch.from_numpy(batch['u'][:, separate_time, :, :]).float().unsqueeze(1).to(device)
  #u = torch.from_numpy(batch['u'][:, separate_time, :, :]).float().unsqueeze(1) / 4.0e-09
  
  #u = torch.zeros_like(u)
  vp = torch.from_numpy(batch['vp']).float().unsqueeze(1)
  u_next = torch.from_numpy(batch['u_labels']) / 4.0e-09

  predictions = torch.zeros_like(u_next).float()

  # x = torch.zeros_like(u_next).float()
  # x[:, :separate_time, :, :] = torch.from_numpy(batch['u'][:, :separate_time, :, :]).float()
  # x = x.to(device)

  predictions[:, :separate_time, :, :] = u_next[:, :separate_time, :, :]




  hidden = vp.to(device)
  context = None

  for t in range(0, nt - 1):
    #for b in range(batch_size):
    #  u[b, :, batch['scri_l'][b], batch['scrj_l'][b]] += batch['q'][b, t] * (batch['dT'][b] / batch['dd'][b])**2 / 4.0e-09
    # print(t)
    if model.model_type == 'GRU':
      pred, _ = model((u).to(device), hidden.to(device))
    elif model.model_type == 'LSTM':
      if t==0:
        pred, _, context = model(u, torch.zeros_like(hidden), hidden)
        
      else:
            # print(solutions.shape, eq_features.shape)
        shape = u.shape
 
        out = model.encoder(seq_to_cnn(u.contiguous().unsqueeze(2)))
   
        x = model.encoder_1(out)
   
        x = x.view(shape[0], shape[1], x.shape[1], x.shape[2], x.shape[3])
        # print(x.shape, context[0][0].shape, context[0][-1].shape)
        # print(x.shape, context[0][0], context[0][-1])
        output, context = model.conv_lstm(x, context[0][0], context[0][-1])
        
        if model.hidden_control is not None:
          h_diff = 0.#self.hidden_loss(output[0][:, :-1, :, :, :], x[:, 1:, :, :, :])
          #h_diff = output[0][:, :-1, :, :, :] - x[:, 1:, :, :, :]
        else:
          h_diff = 0.

        x = output[0]

        # print('here', x.shape)
        # print(seq_to_cnn(x).shape)
        x = model.decoder(seq_to_cnn(x))
        #print('feeee', torch.cat([x, out], dim=1).shape)
        #x = self.decoder_1(torch.cat([x, out], dim=1))
        #x = x + out
        x = model.decoder_1(x) 
        
        #print(x.shape)
        # print(x.shape)
        pred = x.view(shape[0], shape[1], x.shape[1], x.shape[2], x.shape[3]).squeeze(2)

        #context = context[0][-1][-1]

    elif model.model_type == 'AE':
      pred = model((u).to(device))
    
    # predictions[:, t, :, :] =  pred[:, t, :, :]
    # x[:, t, :, :] =  pred[:, t, :, :]
    # u = pred
    #u /= u.std() * predictions[:, t - 1, :, :].std()
    #hidden = u
    #metrix_coeff['correlation'].append(correlation_batch_one_picture(pred.to(device), u_next[:, t, :, :].to(device)).item())
    #metrix_coeff['RMSE'].append(RMSE_batch_one_picture(pred.to(device), u_next[:, t, :, :].to(device)).item())
    # pred_list.append(pred)
    u = torch.from_numpy(batch['u'][:, t, :, :]).float().unsqueeze(1).to(device)
    # hidden = pred
    # if t < separate_time:
    #   continue
    # else:
    #   predictions[:, t, :, :] = pred.squeeze(1)
    # #  u = pred

    

  return predictions, u_next, vp, batch['f0_list']







def test_diff_model__(model, N_min=4, N_max=6, nx=80, nz=80, nt=50, batch_size=2, device='cuda', separate_time=10):

  model.train(False)
  pred_list = []
  # print(nx, nz, nt)
  batch = get_batch(N_min, N_max, nx, nz, nt, batch_size)
  metrix_coeff = {'correlation': [], 'RMSE': []}


  
  factor = 4.0e-09
  X = torch.from_numpy(batch['u']).float()[:, :separate_time, :, :]
  X /= factor
  X_vp = torch.from_numpy(batch['vp']).float()

  X_labels = torch.from_numpy(batch['u_labels']).float()#[:, :separate_time, :, :]
  X_labels /= factor 

  predictions = torch.zeros_like(X_labels).float()
  predictions[:, :separate_time, :, :] = X_labels[:, :separate_time, :, :]


  context = None

  for t in range(0, nt - 1 - separate_time):
    t_nn = 0
    if model.model_type == 'AE':
      start = time.time()
      predictions = model(X[:, :-1, :, :].to(device))
      reg_loss = 0.
      t_nn += (time.time() - start) / batch_size
    else:
      start = time.time()
      print('X ',X.shape)
      pred, reg_loss, _ = model(X[:, :, :, :].to(device), X_vp.unsqueeze(1).to(device))
      t_nn += time.time() - start
      print('p ', pred.shape)
      print('P ', predictions.shape)

    predictions[:, t + separate_time, :, :] = pred[:, -1, :, :]
    # X = predictions[:, :t + separate_time + 1, :, :]
    X = X_labels[:, :t + separate_time + 1, :, :]
    add_source(X, batch)
    print('X1 ',X.shape)
    # predictions[:, t, :, :] =  pred[:, t, :, :]
    # x[:, t, :, :] =  pred[:, t, :, :]
    # u = pred
    #u /= u.std() * predictions[:, t - 1, :, :].std()
    #hidden = u
    #metrix_coeff['correlation'].append(correlation_batch_one_picture(pred.to(device), u_next[:, t, :, :].to(device)).item())
    #metrix_coeff['RMSE'].append(RMSE_batch_one_picture(pred.to(device), u_next[:, t, :, :].to(device)).item())
    # pred_list.append(pred)
    # u = torch.from_numpy(batch['u'][:, t, :, :]).float().unsqueeze(1).to(device)
    # hidden = pred
    # if t < separate_time:
    #   continue
    # else:
    #   predictions[:, t, :, :] = pred.squeeze(1)
    # #  u = pred

    

  return predictions, X_labels, X_vp.unsqueeze(1), batch['f0_list']

def add_source(X, batch):
  tm = X.shape[1]
  for b in range(X.shape[0]):
    X[b, :, batch['scri_l'][b], batch['scrj_l'][b]] += torch.from_numpy(batch['q'][b, :tm] / batch['dd'][b] ** 2 * (batch['dT'][b]) ** 2).float()
    





