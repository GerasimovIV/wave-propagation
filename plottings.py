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


def plot_metrics(model, epoch, loss_hist, epoch_time_nn, epoch_time_fd):
  if epoch !=0:
    print('\nEpoch: {}, train loss: {}, val loss: {}'.format(epoch, loss_hist['train'][-1], loss_hist['val'][-1]))
    print('Time for NN = %f, Time for FD = %f' % (epoch_time_nn[-1], epoch_time_fd[-1]))
    model.compression()
    plt.figure(figsize=(10, 5))
    plt.plot(loss_hist['train'], label='Train')
    plt.plot(loss_hist['val'], label='Val')
    plt.xlabel('#Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend(loc='best')
    plt.show()



def plot_valid(X, predictions, batch_size):
  fig, axs = plt.subplots(3, 4, figsize=(10, 10))
  for c in range(4):
    for s in range(3):
      b_n = np.random.randint(low=0, high=batch_size)
      t_n = np.random.randint(low=int(X.shape[1]*0.5), high=X.shape[1]-1)
      if c==0:  
        axs[s, c].imshow(X[b_n, t_n, 0, :, :])
      elif c==1:
        axs[s, c].imshow(predictions[b_n, t_n, :, :].cpu().data.numpy())
      elif c==2:
        axs[s, c].imshow(X[b_n, t_n , -1, :, :])
      elif c==3:
        axs[s, c].imshow(np.abs(X[b_n, t_n , 0, :, :]-predictions[b_n, t_n, :, :].cpu().data.numpy()))

  plt.show()




