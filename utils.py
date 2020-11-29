from scipy.ndimage import gaussian_filter
import numpy as np
import os
import torch.nn.functional as F
import torch

from time import gmtime, strftime

import torch
from PIL import Image
import torchvision.transforms as transforms

import time

class Simulation(object):
  def __init__(self, nx, nz, dd, nt, dt, srcx, srcz, 
               nabs, a, FreeSurf, vp, wav):
    self.nx = nx # number of grid points in the horizontal direction
    self.nz = nz # number of grid points in the vertical direction

    self.dd = dd # grid cell size
    self.nt = nt # number of time samples to be modeled
    self.dt = dt # time step

    self.srcx = srcx # source horizontal location in meters
    self.srcz = srcz # source vertical location in meters

    self.nabs = nabs # number of absorbing cells on the boundary
    self.a = a # strength of sponge layer
    self.FreeSurf = FreeSurf # free surface condition of top (False for now)

    self.vp = vp
    self.wav = wav
    
    size = (self.vp.shape[0], self.vp.shape[1], self.wav.shape[0])
    self.field2d = np.zeros(size, dtype=np.float) # define variables  - field2d is output wavefield
    self.gen_absorb()
  
  def gen_absorb(self):
    absorb = np.ones((self.nx,self.nz))
    abs_coefs = np.zeros(self.nabs)
    abs_coefs = np.exp(-(self.a**2 * (self.nabs-np.arange(self.nabs))**2))
    absorb[:self.nabs,:] = absorb[:self.nabs,:]*np.expand_dims(abs_coefs,1)
    absorb[-self.nabs:,:] = absorb[-self.nabs:,:]*np.expand_dims(abs_coefs[::-1],1)
    absorb[:,-self.nabs:] = absorb[:,-self.nabs:]*abs_coefs[::-1]
    if(self.FreeSurf==False):
        absorb[:,:self.nabs] = absorb[:,:self.nabs]*abs_coefs
    self.absorb = absorb


  def comp_deriv(self, p):
    pdx2 = np.zeros(p.shape)
    pdz2 = np.zeros(p.shape)

    pdx2[1: -1, 1: -1] = (p[2:, 1: -1] - 2*p[1: -1, 1: -1] + p[: -2, 1: -1]) / (self.dd**2)
    pdz2[1: -1, 1: -1] = (p[1: -1, 2: ] - 2*p[1: -1, 1: -1] + p[1: -1, :-2]) / (self.dd**2)
    return pdx2, pdz2


  def fd_ac(self):
    srci = int(self.srcx / self.dd) # find where source is located on the grid
    srcj = int(self.srcz / self.dd)

    self.srci = srci
    self.srcj = srcj
    
    p = np.zeros((self.nx, self.nz), dtype=np.float) # these are pressures at current, prev and next steps
    ppast = np.zeros((self.nx, self.nz), dtype=np.float)
    pfut = np.zeros((self.nx, self.nz), dtype=np.float)
    
    vp2 = self.vp**2 # square of velocity for easier computation

    for i in range(self.nt): # main loop
        pdx2, pdz2 = self.comp_deriv(p) # compute pressure derivatives
        pfut = 2*p + vp2*self.dt**2 * (pdx2 + pdz2) - ppast # compute future pressure from current and prev 
        
        pfut[srci, srcj] = pfut[srci, srcj] + self.wav[i] / self.dd ** 2 * self.dt ** 2 # inject source term at selected point
        
        p *= self.absorb # apply absorbing mask
        pfut *= self.absorb # apply absorbing mask
        
        self.field2d[:, :, i] = p  # save current pressure in output array
        
        ppast = p # redefine arrays moving to next step
        p = pfut


def rgb2gray(rgb):

    r, g, b = rgb[0,:,:], rgb[1,:,:], rgb[2,:,:]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

marm_size = 141
factor = 1.


def image_loader(image_name, imsize):
    loader = transforms.Compose([transforms.RandomCrop(int(marm_size * factor)),
                             transforms.Resize(imsize),  # scale imported image
                             transforms.ToTensor()])
    
    image = Image.open(image_name)
    image = loader(image)    
    return image

def get_vp(imsize, path_img='./images_vp/'):
  list_img = os.listdir(path_img)
  indx = np.random.randint(low=0, high=len(list_img))
 
  img_tensor = image_loader(path_img+list_img[indx], imsize)
  img_tensor = rgb2gray(img_tensor)
  img_tensor = img_tensor.numpy()
  img_tensor -= img_tensor.min()
  img_tensor = img_tensor / img_tensor.max()

  low = np.random.randint(low=5, high=40)
  high = np.random.randint(low=low + 10, high=100)

  low *= 1e+2
  high *= 1e+2

  img_tensor = img_tensor * (high - low) + low
  img_tensor = np.clip(img_tensor, low, high)

  return img_tensor


class ParamsGenerator(object):
  def __init__(self, 
               N_min, N_max, 
               nx, nz, nt,
               rs=0):
    
    self.rs = rs
    
    self.nx, self.nz, self.nt = nx, nz, nt

    self.vp = get_vp((nz, nz))
    
    self._set_min_max('N', N_min, N_max)
    self._set_min_max('vp', self.vp.min(), self.vp.max())

    self._generate()

  
  def _set_min_max(self, name, var_min, var_max):
    setattr(self, name + '_min', var_min)
    setattr(self, name + '_max', var_max)

  
  def _generate(self):
    self.N_lam = np.random.uniform(low=self.N_min, high=self.N_max)

    self.f0 = np.random.uniform(low=6., high=256.)
    
    f_max = self.f0 * 2.5

    dd = self.vp_min / self.N_lam / f_max
    self.dd = np.random.uniform(low=dd * 0.7, high=dd * 0.9)

    dt = self.dd/self.vp_max/np.sqrt(2.)

    self.dt = np.random.uniform(low=dt * 0.7, high=dt * 0.9)

  
  def set_simulator_params(self):
    nabs = int(min(self.nx, self.nz) * 0.1)

    self.srcx = self._get_rand_src(self.dd, self.nx, nabs)
    self.srcz = self._get_rand_src(self.dd, self.nz, nabs)
    
    simulator = Simulation(self.nx, self.nz, self.dd, self.nt, self.dt,
                           self.srcx, self.srcz, nabs,
                           self.get_rand_a(),
                           True, self.vp,
                           self.get_rand_wav(self.nt, self.dt, self.f0))
    start = time.time()
    simulator.fd_ac()
    end = time.time()
    return simulator, end - start

  
  def _get_rand_src(self, dd, n, nabs):
    return np.random.uniform(low = (dd * n) * (nabs / n + 0.05), high = (dd * n) * (1 - nabs / n - 0.05))

  
  def get_rand_wav(self, nt, dt, f0):
    time_vec = np.linspace(0, nt * dt, nt) # time vector
    t0 = 1./f0
    scale = 1.
    wav  = scale * (1.0-2.0*np.power(np.pi*f0*(time_vec-t0),2))*np.exp(-np.power(np.pi*f0*(time_vec-t0),2)) # computing the wavelet
    return wav

  
  def get_rand_a(self, ):
    #sigma = np.random.choice(np.array([10, 15, 20]))
    return 0.0053#1. / (2 * sigma**2)



def RMSE_(x, y):
  return torch.sqrt(F.mse_loss(x / x.std(), y / y.std()))

def RMSE_batch(x, y):
  cost = torch.zeros(x.shape[:2])
  for i in range(cost.shape[0]):
    for j in range(cost.shape[1]):
      cost[i, j] = RMSE_(x[i, j], y[i, j])
  return cost.mean()

def RMSE_batch_obo(x, y):
  cost = torch.zeros(x.shape[0])
  for i in range(cost.shape[0]):
    cost[i] = RMSE_(x[i, :], y[i, :])
  return cost.mean()

def correlation_(x, y):
  x /= x.std()
  y /= y.std()
  vx = x - torch.mean(x, axis=(-1, -2))
  vy = y - torch.mean(y, axis=(-1, -2))
  cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
  return cost

def correlation_batch(x, y):
  cost = torch.zeros(x.shape[:2])
  for i in range(cost.shape[0]):
    for j in range(cost.shape[1]):
      cost[i, j] = correlation_(x[i, j], y[i, j])
  return cost.mean()

def correlation_obo(x, y):
  x /= x.std()
  y /= y.std()
  vx = x - torch.mean(x)
  vy = y - torch.mean(y)
  cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
  return cost


def correlation_batch_obo(x, y):
  cost = torch.zeros(x.shape[0])
  for i in range(cost.shape[0]):
    cost[i] = correlation_obo(x[i], y[i])
  return cost.mean()



def save_model(epoch, model, optimizer, loss_hist, metrix_coeff, best_names=[], KI=False, add_name=''):
  state = {'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(), 'loss_hist': loss_hist, 'metrix_coeff': metrix_coeff}
  file_name = './checkpoints/' + add_name + model.__class__.__name__ + '.pth'
  if best_names!=[]:  
    for i in range(len(best_names)):
      file_name = file_name[:-4] + '_best_' + best_names[i] + file_name[-4:]
  elif KI == True:
    file_name = file_name[:-4] + '_KeyboardInterrupt_' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + file_name[-4:]
  torch.save(state, file_name)

  print('Parameters of the model and optimizer are saved to file --> ' + file_name)


def model_best(loss_hist, metrix_coeff):
# find were our model is the best

  result = []
  # print(loss_hist, metrix_coeff)
  for k in loss_hist:
    if loss_hist[k][-1] <= min(loss_hist[k]):
      result.append(k)
  
  if abs(metrix_coeff['correlation'][-1]) >= max(map(abs, metrix_coeff['correlation'])):
    result.append('correlation')

  if metrix_coeff['RMSE'][-1] <= min(metrix_coeff['RMSE']):
    result.append('RMSE')

  return result



def load_model(PATH, model, optimizer, loss_hist, metrix_coeff):
  
  checkpoint = torch.load(PATH)
  
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']
  loss_hist = checkpoint['loss_hist']
  metrix_coeff = checkpoint['metrix_coeff']
  # model_state_dict = {}
  # for key in checkpoint['model_state_dict']:
  #   try:
  #       model_state_dict[key] = checkpoint['model_state_dict'][key]
  #       print(key)
  #   except KeyError:
  #       print('Warning=====================================\nThere is no parameter with key: '
  #             +str(key)+'\nMaybe this is old model\n============================================')
  # model.load_state_dict(model_state_dict)
  model.load_state_dict(checkpoint['model_state_dict'])
  
  return loss_hist, metrix_coeff, epoch
  # model.eval()
  # # - or -
  # model.train() DON'T FORGET AFTER THAT DO IT


def load_metrics(PATH):
  
  checkpoint = torch.load(PATH)
  
  epoch = checkpoint['epoch']
  loss_hist = checkpoint['loss_hist']
  metrix_coeff = checkpoint['metrix_coeff']

  return loss_hist, metrix_coeff, epoch

