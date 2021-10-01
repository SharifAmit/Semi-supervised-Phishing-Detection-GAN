import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer 
import string
import numpy as np
from keras.utils import to_categorical
import argparse

def tokenizer(alphabet,url_length=200):
    dictionary_size = len(alphabet) + 1
    url_shape = (url_length, dictionary_size)
    dictionary = {}
    reverse_dictionary = {}
    for i, c in enumerate(alphabet):
        dictionary[c]=i+1
        reverse_dictionary[i+1]=c
    return dictionary, reverse_dictionary
    
def data_npz(good,bad,alphabet,dictionary,samples,url_length):
        good_data = []   
        i = 0 
        for i in range(26000):
            line = good['URL'][i]
            this_sample=np.zeros(url_shape)

            line = line.lower()
            if len ( set(line) - set(alphabet)) == 0 and len(line) < args.url_length:
                for i, position in enumerate(this_sample):
                    this_sample[i][0]=1.0

                for i, char in enumerate(line):
                    this_sample[i][0]=0.0
                    this_sample[i][dictionary[char]]=1.0
                good_data.append(this_sample)
            else:
                print("Uncompatible line:",  line)
            
        #print("Data ready. Lines:", len(good_data))
        good_data = np.array(good_data)
        good_data = good_data[:samples]
        print ("Array Shape:", good_data.shape)

        bad_data = []   
        i = 0 
        for i in range(30000):
            line = bad['URL'][i]
            this_sample=np.zeros(url_shape)

            line = line.lower()
            if len ( set(line) - set(alphabet)) == 0 and len(line) < args.url_length:
                for i, position in enumerate(this_sample):
                    this_sample[i][0]=1.0

                for i, char in enumerate(line):
                    this_sample[i][0]=0.0
                    this_sample[i][dictionary[char]]=1.0
                bad_data.append(this_sample)
            else:
                print("Uncompatible line:",  line)
            
        #print("Data ready. Lines:", len(bad_data))
        bad_data = np.array(bad_data)
        bad_data = bad_data[:samples]

        x_train_len = int(samples* 0.8)
        x_train = np.concatenate((good_data[:x_train_len,:,:], bad_data[:x_train_len,:,:]),axis=0)
        x_test = np.concatenate((good_data[x_train_len:samples,:,:], bad_data[x_train_len:samples,:,:]),axis=0)

        good_label = np.ones((samples,1))
        bad_label = np.zeros((samples,1))
        y_train = np.concatenate((good_label[:x_train_len,:], bad_label[:x_train_len,:]),axis=0)
        y_train_cat = to_categorical(y_train)
        y_test = np.concatenate((good_label[x_train_len:samples,:], bad_label[x_train_len:samples,:]),axis=0)
        y_test_cat = to_categorical(y_test)

        np.savez_compressed('phishing.npz', X_train=x_train, X_test=x_test, y_train=y_train_cat, y_test=y_test_cat)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--url_length', type=int, default=200)
    parser.add_argument('--npz_filename', type=str, default='phishing.npz')
    parser.add_argument('--n_sampels',types=int, default=50000,help='number of good and bad samples.')
    args = parser.parse_args()

    
    alphabet = string.ascii_lowercase + string.digits + "!#$%&()*+,-./:;<=>?@[\\]^_`{|}~"
    dictionary_size = len(alphabet) + 1
    url_shape = (args.url_length, dictionary_size)

    df = pd.read_csv('phishing_site_urls.csv')
    good = df[df['Label']=='good']
    bad = df[df['Label']=='bad']
    good.reset_index(drop=True, inplace=True)
    bad.reset_index(drop=True, inplace=True)


    each_class_samples= args.n_samples //2
    dictionary, reverse_dictionary = tokenizer(alphabet,url_length= args.url_length)

    data_npz(good,bad,alphabet,dictionary,samples=each_class_samples,url_length=args.length)

    

