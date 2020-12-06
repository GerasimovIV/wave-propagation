import numpy as np
import matplotlib.pyplot as plt


def show_grig(img_batch, n_rows):
    batch_size = img_batch.shape[0]
    
    assert batch_size == n_rows ** 2
    
    plt.figure(figsize=(20, 20))
    for i in range(batch_size):
        plt.subplot(n_rows, n_rows, i + 1)
        plt.imshow(img_batch[i], cmap=plt.cm.BuPu, interpolation='nearest')
        plt.axis('off')
        plt.colorbar()
    plt.show()
  

def plots():
  plt.figure(figsize=(10, 5))
  plt.plot(metric[key], label=key)
  plt.xlabel(xl)
  plt.ylabel(yl)
  plt.legend(loc='best')
  plt.title(title)

def plot_metric(metric, xl, yl, title, on_one_plot=True):
  if on_one_plot==True:
    plt.figure(figsize=(10, 5))
    for key in metric:
      plt.plot(metric[key], label=key)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.legend(loc='best')
    plt.title(title)
  else:
    for key in metric:
      plt.figure(figsize=(10, 5))
      plt.plot(metric[key], label=key)
      plt.xlabel(xl[key])
      plt.ylabel(yl[key])
      if key == 'RMSE':
        plt.yscale('log')
      plt.legend(loc='best')
      plt.title(title[key])
  

  plt.show()
  

def plot_metrics(model, epoch, loss_hist, metrix_coeff):
  if epoch !=0:
    print('\nEpoch: {}, train loss: {}, val loss: {}'.format(epoch, loss_hist['train'][-1], loss_hist['val'][-1]))
    # print('Time for NN = %f, Time for FD = %f' % (epoch_time_nn[-1], epoch_time_fd[-1]))
    try:
      model.compression()
    except ModuleAttributeError:
      print('Model have not compression')
    
    plot_metric(loss_hist, '#Epochs', 'Loss', 'Loss')
    plot_metric(metrix_coeff, {'RMSE': '#Epochs', 'correlation': '#Epochs'}, 
                              {'RMSE': '', 'correlation': ''}, 
                              {'RMSE': 'Metric RMSE', 'correlation': 'Metric correlation'}, on_one_plot=False)
    


def plot_metrics_obo(model, epoch, loss_hist, metrix_coeff):
  if epoch !=0:
    print('\nEpoch: {}, train loss: {}, val loss: {}'.format(epoch, loss_hist['train'][-1], loss_hist['val'][-1]))
    # print('Time for NN = %f, Time for FD = %f' % (epoch_time_nn[-1], epoch_time_fd[-1]))

    
    plot_metric(loss_hist, '#Epochs', 'Loss', 'Loss')
    plot_metric(metrix_coeff, {'RMSE': '#Epochs', 'correlation': '#Epochs'}, 
                              {'RMSE': '', 'correlation': ''}, 
                              {'RMSE': 'Metric RMSE', 'correlation': 'Metric correlation'}, on_one_plot=False)
    


def plot_valid(X, predictions, X_vp, batch_size, f0_list):
  fig, axs = plt.subplots(3, 4, figsize=(20, 20))
  for s in range(3):
    t_n = np.random.randint(low=0, high=X.shape[1] - 1)
    b_n = np.random.randint(low=0, high=batch_size)
    # b_n = 0
    # t_n = -1
    for c in range(4):
      if c==0:  
        im = axs[s, c].imshow(X[b_n, t_n, :, :], cmap=plt.cm.BuPu)
        axs[s, c].set(title='sol '+'t: '+str(t_n)+' b_n: '+str(b_n)+' f0 :'+str(round(f0_list[b_n], 2)))
        fig.colorbar(im, ax=axs[s, c])
      elif c==1:
        im = axs[s, c].imshow(predictions[b_n, t_n, :, :].cpu().data.numpy(), cmap=plt.cm.BuPu)
        axs[s, c].set(title='pred '+'t: '+str(t_n)+' b_n: '+str(b_n)+' f0 :'+str(round(f0_list[b_n], 2)))
        fig.colorbar(im, ax=axs[s, c])
      elif c==2:
        im = axs[s, c].imshow(X_vp[b_n, :, :], cmap=plt.cm.BuPu)
        axs[s, c].set(title='vp '+'t: '+str(t_n)+' b_n: '+str(b_n)+' f0 :'+str(round(f0_list[b_n], 2)))
        fig.colorbar(im, ax=axs[s, c])
      elif c==3:
        im = axs[s, c].imshow(np.abs(X[b_n, t_n, :, :]-predictions[b_n, t_n, :, :].cpu().data.numpy()), cmap=plt.cm.BuPu)
        axs[s, c].set(title='dif '+'t: '+str(t_n)+' b_n: '+str(b_n)+' f0 :'+str(round(f0_list[b_n], 2)))
        fig.colorbar(im, ax=axs[s, c])

  plt.show()



def plot_test_video(X_labels, predictions, vp, f0_list):

  b_n = 0
  print(type(predictions))
  for t in range(predictions.shape[1]):
    fig, axs = plt.subplots(1, 4, figsize=(20, 10))

    im = axs[0].imshow(X_labels[b_n, t, :, :], cmap=plt.cm.BuPu)
    axs[0].set(title='sol '+'t: '+str(t)+' b_n: '+str(b_n)+' f0 :'+str(round(f0_list[b_n], 2)))
    fig.colorbar(im, ax=axs[0])

    im = axs[1].imshow(predictions[b_n, t, :, :].cpu().data.numpy(), cmap=plt.cm.BuPu)
    axs[1].set(title='pred '+'t: '+str(t)+' b_n: '+str(b_n)+' f0 :'+str(round(f0_list[b_n], 2)))
    fig.colorbar(im, ax=axs[1])

    im = axs[2].imshow(vp[b_n, 0, :, :].cpu().data.numpy(), cmap=plt.cm.BuPu)
    axs[2].set(title='vp '+'t: '+str(t)+' b_n: '+str(b_n)+' f0 :'+str(round(f0_list[b_n], 2)))
    fig.colorbar(im, ax=axs[2])

    im = axs[3].imshow(X_labels[b_n, t, :, :]-predictions[b_n, t, :, :].cpu().data.numpy(), 
                       cmap=plt.cm.BuPu)
    axs[3].set(title='diff '+'t: '+str(t)+' b_n: '+str(b_n)+' f0 :'+str(round(f0_list[b_n], 2)))
    fig.colorbar(im, ax=axs[3])

    plt.show()


def plot_metrics(model, epoch, loss_hist, metrix_coeff):
  if epoch !=0:
    print('\nEpoch: {}, train loss: {}, val loss: {}'.format(epoch, loss_hist['train'][-1], loss_hist['val'][-1]))
    # print('Time for NN = %f, Time for FD = %f' % (epoch_time_nn[-1], epoch_time_fd[-1]))
    try:
      model.compression()
    except ModuleAttributeError:
      print('Model have not compression')
    
    plot_metric(loss_hist, '#Epochs', 'Loss', 'Loss')
    plot_metric(metrix_coeff, {'RMSE': '#Epochs', 'correlation': '#Epochs'}, 
                              {'RMSE': '', 'correlation': ''}, 
                              {'RMSE': 'Metric RMSE', 'correlation': 'Metric correlation'}, on_one_plot=False)
    


def plot_result_test(results_test, model):

  metrix_coeff, predictions, Labels_u, batch = results_test

  try:
    model.compression()
  except ModuleAttributeError:
    print('Model have not compression')

  plot_metric(metrix_coeff, {'RMSE': '#', 'correlation': '#'}, 
                              {'RMSE': '', 'correlation': ''}, 
                              {'RMSE': 'Metric RMSE', 'correlation': 'Metric correlation'}, on_one_plot=False)
  plot_test(Labels_u, predictions, batch['u_vp'][:,0,  1, :, :], 
            batch_size=Labels_u.shape[0], f0_list=batch['f0_list'])
