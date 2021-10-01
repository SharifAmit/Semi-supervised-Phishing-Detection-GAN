
import tensorflow as tf
from keras.layers import Layer, Reshape, Activation, Conv1D, Conv1DTranspose, Dropout
from keras.layers import Input, Add, Concatenate, Embedding,LeakyReLU,Dense, BatchNormalization, Flatten
from keras.optimizers import Adam
from keras.models import Model
from keras.initializers import RandomNormal
from functools import partial


# define the standalone discriminator model
def define_discriminator(in_shape=(200,67), n_classes=2):
    # weight initialization
    #init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=(in_shape))    
    # downsample to 14x14
    fe = Conv1D(16, 3, strides=1, padding='same')(in_image)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv1D(16, 3, strides=2, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)
    # normal
    fe = Conv1D(32, 3, strides=1, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv1D(32, 3, strides=2, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # downsample to 7x7
    fe = Conv1D(128, 3, strides=1, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv1D(128, 3, strides=2, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)
   
    # flatten feature maps
    fe = Flatten()(fe)
    dense_1 = Dense(256)(fe)
    dense_1 = LeakyReLU()(dense_1)
    dense_1 = Dense(64)(dense_1)
    # real/fake output
    out1 = Dense(1, activation='sigmoid')(fe)
    # class label output
    out2 = Dense(n_classes, activation='softmax')(fe)
    # define model
    model = Model(in_image, [out1, out2],name="Discriminator")
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['mse', 'categorical_crossentropy'], optimizer=opt)
    model.summary()
    return model

# define the standalone generator model
def define_generator(latent_dim=(50,),signal_shape=(200,67), label_shape=(2,)):
    # weight initialization
    #init = RandomNormal(stddev=0.02)
    depth = 4 #32
    dropout = 0.25
    dim = signal_shape[0] #
    
    # signal_input
    in_signal = Input(shape=signal_shape)
    si = in_signal
    #si = Reshape((280,1))(in_signal)
    
    # label input
    in_label = Input(shape=label_shape)
    # embedding for categorical input
    li = Embedding(2, 50)(in_label)
    # linear multiplication
    n_nodes = 200 * 1
    li = Dense(n_nodes)(li)
    # reshape to additional channel
    li = Reshape((200,2))(li)
    
    # noise  input
    in_lat = Input(shape=latent_dim)
    lat = Reshape((1,50))(in_lat)
    # foundation for 7x7 image
    n_nodes = dim*depth
    gen = Dense(n_nodes)(lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((dim, depth))(gen)
    # merge image gen and label input
    merge = Concatenate()([gen, li,si]) #gen=200,32 x li=200,2 x si=200,67 ## Uncomment this
    #merge = Concatenate()([gen, li]) #gen=280,32 li=280,5
 

    gen = Conv1D(32, 3, strides=1, padding='same')(merge)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv1D(32, 3, strides=2, padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv1D(64, 3, strides=1, padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv1D(64, 3, strides=2, padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv1D(128, 3, strides=1, padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv1D(128, 3, strides=2, padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv1DTranspose(128, 3, strides=2, padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv1DTranspose(64, 3, strides=2, padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv1DTranspose(32, 3, strides=2, padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)


    #gen = Reshape((200,67))(gen)

    gen = Conv1D(67, 3, strides=1, padding='same')(gen)
    out_layer = Activation('sigmoid')(gen)

    model = Model([in_signal,in_lat, in_label], out_layer,name="Generator")
    #model = Model([in_lat, in_label], out_layer,name="Generator")
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='mse', optimizer=opt)
    model.summary()
    return model

def define_gan(g_model, d_model,latent_dim=(200,67), signal_shape=(200,67),label_shape=(2,)):
    #in_signal = Input(shape=signal_shape)
    #in_label = Input(shape=label_shape)
    #in_lat = Input(shape=latent_dim)
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect the outputs of the generator to the inputs of the discriminator
    [out1,out2] = d_model(g_model.output)
    # define gan model as taking noise and label and outputting real/fake and label outputs
    #model = Model(g_model.input, gan_output)
    model = Model([g_model.input[0],g_model.input[1],g_model.input[2]],[out1,out2])
    #model = Model([g_model.input[0],g_model.input[1]],[out1,out2])
    #model = Model([in_signal,in_lat, in_label],[out1,out2])
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['mse', 'categorical_crossentropy'], optimizer=opt,loss_weights=[1,10])
    model.summary()
    return model