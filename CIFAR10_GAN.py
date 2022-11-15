import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, ReLU, Dense, LeakyReLU, conv2D,  BatchNormalization, ReLU
from keras.layers.core import Activation, Reshape, Conv2DTranspose
from keras import initializers
from keras.layers.convolutional import Conv2D
from keras.layers.core import Flatten
from keras.optimizers import SGD, Adam
from keras import initializers
from tensorflow.keras.layers import Dropout
import matplotlib.pylab as plt
import os
# from main import *

# We need to create a class which contains both the genrator and discriminator models so that we can train them together


def resize_image(X):
    Y = np.float32(X)
    Y = (Y/255-0.5)*2
    Y = np.clip(Y, -1, 1)
    return Y


def generator_architecture(input_shape=(32, 32, 3)):
    latent_dim = 512
    init = initializers.RandomNormal(stddev=0.02)
    generator = Sequential()
    generator.add(Dense(2*2*512, input_shape=(latent_dim,),
                        kernel_initializer=init))
    generator.add(Reshape((2, 2, 512)))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))
    generator.add(Conv2DTranspose(
        256, kernel_size=5, strides=2, padding='same'))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))
    generator.add(Conv2DTranspose(
        128, kernel_size=5, strides=2, padding='same'))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))
    generator.add(Conv2DTranspose(
        64, kernel_size=5, strides=2, padding='same'))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))
    generator.add(Conv2DTranspose(3, kernel_size=5, strides=2, padding='same',
                                  activation='tanh'))
    return generator


def discriminator_architecture(img_shape=(32, 32, 3)):
    discriminator = Sequential()
    discriminator.add(Conv2D(64, kernel_size=5, strides=2, padding='same',
                             input_shape=(img_shape), kernel_initializer=init))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Conv2D(256, kernel_size=5, strides=2, padding='same'))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Conv2D(512, kernel_size=5, strides=2, padding='same'))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation='sigmoid'))
    return discriminator


def train(class_name, train_X):
    discriminator = discriminator_architecture()
    discriminator.compile(Adam(lr=0.0003, beta_1=0.5), loss='binary_crossentropy',
                          metrics=['binary_accuracy'])
    discriminator.trainable = False
    z = Input(shape=(512,))
    generator = generator_architecture()
    img = generator(z)
    decision = discriminator(img)
    GAN = Model(inputs=z, outputs=decision)
    GAN.compile(Adam(lr=0.0004, beta_1=0.5), loss='binary_crossentropy',
                metrics=['binary_accuracy'])
    # Check if there is a file class_name+"GAN.h5" and load it
    if os.path.isfile(class_name+"_GAN.h5"):
        generator.load_weights(class_name+"_GAN.h5")
        return GAN
    # Train the GAN
    EPOCHS = 5000
    BATCH_SIZE = 32
    SMOOTH = 0.1
    X_train = resize_image(train_X)

    real = np.ones((BATCH_SIZE, 1))
    fake = np.zeros((BATCH_SIZE, 1))
    d_loss = []
    g_loss = []
    for e in range(EPOCHS+1):
        for i in range(len(X_train)//BATCH_SIZE):
            # Train the discriminator
            discriminator.trainable = True
            # On real samples
            X_batch = X_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            d_loss_real = discriminator.train_on_batch(
                x=X_batch, y=real * (1 - SMOOTH))
            # On fake samples
            z = np.random.normal(0, 1, size=(BATCH_SIZE, 512))
            X_fake = generator.predict_on_batch(z)
            d_loss_fake = discriminator.train_on_batch(x=X_fake, y=fake)
            d_loss_batch = 0.5 * np.add(d_loss_real[0], d_loss_fake[0])
            # Train the generator
            discriminator.trainable = False
            g_loss_batch = GAN.train_on_batch(x=z, y=real)
        if e % 100 == 0:
            print("Epoch: {}/{} Discriminator Loss: {} Generator Loss: {}".format(
                e, EPOCHS, d_loss_batch, g_loss_batch))
            d_loss.append(d_loss_batch)
            g_loss.append(g_loss_batch)
    # Save the weights
    generator.save_weights(class_name+"_GAN.h5")
    return generator