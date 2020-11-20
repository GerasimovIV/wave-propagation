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
      plt.legend(loc='best')
      plt.title(title[key])
  

  plt.show()
  

def plot_metrics(model, epoch, loss_hist, epoch_time_nn, epoch_time_fd, metrix_coeff):
  if epoch !=0:
    print('\nEpoch: {}, train loss: {}, val loss: {}'.format(epoch, loss_hist['train'][-1], loss_hist['val'][-1]))
    print('Time for NN = %f, Time for FD = %f' % (epoch_time_nn[-1], epoch_time_fd[-1]))
    model.compression()
    
    plot_metric(loss_hist, '#Epochs', 'Loss', 'Loss')
    plot_metric(metrix_coeff, {'RMSE': '#Epochs', 'correlation': '#Epochs'}, 
                              {'RMSE': '', 'correlation': ''}, 
                              {'RMSE': 'Metric RMSE', 'correlation': 'Metric correlation'}, on_one_plot=False)
    
    # plt.figure(figsize=(10, 5))
    # plt.plot(loss_hist['train'], label='Train')
    # plt.plot(loss_hist['val'], label='Val')
    # plt.xlabel('#Epochs')
    # plt.ylabel('Loss')
    # plt.title('Loss')
    # plt.legend(loc='best')
    # plt.show()

    # plt.figure(figsize=(10, 5))
    # plt.plot(metrix_coeff['correlation'], label='Correlation')
    # plt.plot(metrix_coeff['RMSE'], label='RMSE')
    # plt.xlabel('#Epochs')
    # # plt.ylabel('Loss')
    # plt.title('Metrics')
    # plt.legend(loc='best')
    # plt.show()



def plot_valid(X, predictions, batch_size):
  fig, axs = plt.subplots(3, 4, figsize=(20, 20))
  for s in range(3):
    t_n = np.random.randint(low=0, high=X.shape[1]-1)
    b_n = np.random.randint(low=0, high=batch_size)
    b_n = 0
    # t_n = -1
    for c in range(4):
      if c==0:  
        im = axs[s, c].imshow(X[b_n, t_n, 0, :, :], cmap=plt.cm.BuPu)
        axs[s, c].set(title='sol '+'t: '+str(t_n)+' b_n: '+str(b_n))
        fig.colorbar(im, ax=axs[s, c])
      elif c==1:
        im = axs[s, c].imshow(predictions[b_n, t_n, :, :].cpu().data.numpy(), cmap=plt.cm.BuPu)
        axs[s, c].set(title='pred '+'t: '+str(t_n)+' b_n: '+str(b_n))
        fig.colorbar(im, ax=axs[s, c])
      elif c==2:
        im = axs[s, c].imshow(X[b_n, t_n , -1, :, :], cmap=plt.cm.BuPu)
        axs[s, c].set(title='vp '+'t: '+str(t_n)+' b_n: '+str(b_n))
        fig.colorbar(im, ax=axs[s, c])
      elif c==3:
        im = axs[s, c].imshow(np.abs(X[b_n, t_n , 0, :, :]-predictions[b_n, t_n, :, :].cpu().data.numpy()), cmap=plt.cm.BuPu)
        axs[s, c].set(title='dif '+'t: '+str(t_n)+' b_n: '+str(b_n))
        fig.colorbar(im, ax=axs[s, c])

  plt.show()




