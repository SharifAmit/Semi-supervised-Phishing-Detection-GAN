import numpy as np
from numpy import load
import tensorflow as tf
from numpy.random import randint
from numpy import zeros, ones
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d


# generate points in latent space as input for the generator

def generate_latent_points(latent_dim, n_samples):
  
    x_input = gaussian_filter1d(np.random.randint(high=1.0, low=0.0,size=(latent_dim*n_samples)),4)
    z_input = x_input.reshape(n_samples, latent_dim)

    return z_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, real_sample, labels_input, latent_dim, n_samples):
    z_input = generate_latent_points(latent_dim, n_samples)
    images = generator.predict([real_sample,z_input, labels_input])
    y = zeros((n_samples, 1))
    return [images, labels_input], y

def generate_real_random(dataset, n_samples):
    images, labels = dataset
    ix = randint(0, images.shape[0], n_samples)
    X, labels = images[ix], labels[ix]
    y = ones((n_samples, 1))
    return [X, labels], y

def generate_real_samples(dataset, batch_id, n_samples):
    images, labels = dataset
    start = batch_id*n_samples
    end = start+n_samples
    X, labels = images[start:end], labels[start:end]
    y = ones((n_samples, 1))
    return [X, labels], y