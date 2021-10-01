from src.dataloader import *
from src.log_and_visualization import *
from src.model import *
import time
import os
import string 
import gc
import keras.backend as K

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=200, n_batch=64,savedir="dummy"):
    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    print('batch per epoch: %d' % bat_per_epo)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    print('number of steps: %d' % n_steps)
    dr1_hist, dr2_hist, df1_hist, df2_hist =  list(),list(), list(), list()
    g_hist, gan_hist =  list(), list()

    alphabet = string.ascii_lowercase + string.digits + "!#$%&()*+,-./:;<=>?@[\\]^_`{|}~" 
    reverse_dictionary = {}
    for i, c in enumerate(alphabet):
        reverse_dictionary[i+1]=c
    # calculate the size of half a batch of samples
    #half_batch = int(n_batch / 2)
    # manually enumerate epochs
    b = 0
    start_time = time.time()
    for e in range(n_epochs):
      for i in range(bat_per_epo):
          for j in range(2):
              d_model.trainable = True
              g_model.trainable = False
              # get randomly selected 'real' samples
              [X_real, labels_real], y_real = generate_real_samples(dataset, i, n_batch)
              # update discriminator model weights
            # print(X_real.shape,y_real.shape,labels_real[1].shape)
              
              _,d_r1,d_r2 = d_model.train_on_batch(X_real, [y_real, labels_real])
              # generate 'fake' examples
              [X_fake, labels_fake], y_fake = generate_fake_samples(g_model,X_real, labels_real,latent_dim, n_batch)
              # update discriminator model weights
              _,d_f1,d_f2 = d_model.train_on_batch(X_fake, [y_fake, labels_fake])
          # update the generator via the discriminator's error
          d_model.trainable = False
          g_model.trainable = True
          # prepare points in latent space as input for the generator
          z_input = generate_latent_points(latent_dim, n_batch)
          # Sample batch of data
          [X_real, labels_real], y_real = generate_real_samples(dataset, i, n_batch)
          g_loss = g_model.train_on_batch([X_real, z_input,labels_real],X_real)
          gan_loss,_,_ = gan_model.train_on_batch([X_real, z_input, labels_real], [y_real, labels_real])
          # summarize loss on this batch
          print('>%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f], gan[%.3f]' % (i+1, d_r1,d_r2, d_f1,d_f2, g_loss,gan_loss))
          dr1_hist.append(d_r1) 
          dr2_hist.append(d_r2) 
          df1_hist.append(d_f1) 
          df2_hist.append(d_f2)
          g_hist.append(g_loss)
          gan_hist.append(gan_loss)

          # evaluate the model performance every 'epoch'
          #if (i+1) % (bat_per_epo * 1) == 0:
      summarize_performance_fixed(reverse_dictionary,b,g_model,d_model, dataset, 3,latent_dim, savedir=savedir)
      b = b + 1
      per_epoch_time = time.time()
      total_per_epoch_time = (per_epoch_time - start_time)/3600.0
      print(total_per_epoch_time)
            #summarize_performance(i, g_model, latent_dim,X_real,n_samples=n_batch,savedir=savedir)
    plot_history(dr1_hist, dr2_hist, df1_hist, df2_hist, g_hist, gan_hist,savedir=savedir)        
    to_csv(dr1_hist, dr2_hist, df1_hist, df2_hist, g_hist, gan_hist,savedir=savedir)