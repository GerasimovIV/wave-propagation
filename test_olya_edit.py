import torch
from train_utils import get_batch

def test(model, N_min=4, N_max=6, nx=80, nz=80, nt=50, batch_size=2, device='cuda'):

  model.train(False)
  pred_list = []
  batch = get_batch(N_min, N_max, nx, nz, nt, batch_size)

  # u = torch.from_numpy(batch['u_vp'][:, 10, 0, :, :]).float().unsqueeze(1)
  u = torch.from_numpy(batch['u_vp'][:, :-1, 0, :, :]).float() / 4.0e-09
  
  #u = torch.zeros_like(u)
  vp = torch.from_numpy(batch['u_vp'][:, 0, 1, :, :]).float().unsqueeze(1)
  u_next = torch.from_numpy(batch['u_labels']) / 4.0e-09

  predictions = torch.zeros_like(u_next[:, 10:, :, :])

  # for t in range(0, nt - 10 - 1):

    # for b in range(batch_size):
    #   u[b, :, batch['scri_l'][b], batch['scrj_l'][b]] += batch['q'][b, t] * (batch['dT'][b] / batch['dd'][b])**2 / 4.0e-09
    
    # print(u.shape)
  pred, _ = model((u[:, :, :, :]).to(device), vp.to(device))

    # metrix_coeff['correlation'].append(correlation_batch_one_picture(pred.to(device), u_next[:, i, :, :].to(self.device)).item())
    # metrix_coeff['RMSE'].append(RMSE_batch_one_picture(pred.to(device), u_next[:, i, :, :].to(device)).item())
    # pred_list.append(pred)
    # print(predictions[:, t, :, :].shape, pred.shape)
  predictions = pred
    #print((predictions[:, t, :, :] == u.squeeze(1)).sum())
    # u = pred

  return predictions, u_next, vp, batch['f0_list']
  

