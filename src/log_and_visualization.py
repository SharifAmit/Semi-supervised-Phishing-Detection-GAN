import matplotlib.pyplot as plt
import pandas as pd 
import os
from src.dataloader import *
import random

def summarize_performance_fixed(reverse_dictionary, step,g_model,d_model ,dataset, n_samples=3,latent_dim=128,  savedir='weights_plots'):
    if not os.path.exists(savedir):
            os.makedirs(savedir)
    # select a sample of input images
    [X,labels],_ = generate_real_random(dataset, n_samples)
    # generate a batch of fake samples
    [X_fake, _], _ = generate_fake_samples(g_model,X, labels, latent_dim, n_samples)

    for url in X_fake:
                this_url_gen = ""
                for position in url:
                    this_index = np.argmax(position)
                    if this_index != 0:
                        this_url_gen += reverse_dictionary[this_index]

                print(this_url_gen)
    # save the generator model
    filename2 = savedir+'/'+'gmodel_%06d.h5' % (step+1)
    g_model.save(filename2)
    filename3 = savedir+'/'+'dmodel_%06d.h5' % (step+1)
    d_model.save(filename3)
    print('>Saved: %s and %s' % (filename2, filename3))


def to_csv(dr1_hist, dr2_hist, df1_hist, df2_hist, g_hist, gan_hist,savedir='dummy'):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    dr1 = np.array(dr1_hist)
    dr2 = np.array(dr2_hist)
    df1 = np.array(df1_hist)
    df2 = np.array(df2_hist)
    g = np.array(g_hist)
    gan = np.array(gan_hist)
    df = pd.DataFrame(data=(dr1,dr2,df1,df2,g,gan)).T
    df.columns=["dr1","dr2","df1","df2","g","gan"]
    filename = savedir+"/ecg-atk-loss.csv"
    df.to_csv(filename)

def plot_history(dr1_hist, dr2_hist, df1_hist, df2_hist, g_hist, gan_hist,savedir='dummy'):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    plt.plot(dr1_hist, label='dr1')
    plt.plot(dr2_hist, label='dr2')
    plt.plot(df1_hist, label='dfm1')
    plt.plot(df1_hist, label='dfm2')
    plt.plot(g_hist, label='g_loss')
    plt.plot(gan_hist, label='gan_loss')
    plt.legend()
    filename = savedir+'/plot_line_plot_loss.png'
    plt.savefig(filename)
    plt.close()
    print('Saved %s' % (filename))