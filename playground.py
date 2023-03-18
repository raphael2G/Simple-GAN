import tensorflow as tf
import numpy as np
from matplotlib import pyplot

# Steps to creating and training our own GAN utilizing the MNIST Dataset

# Create Generator Network 
# this will take in random noise of size (64, 1), and output generated data of size (784,1)
generator = tf.keras.Sequential([
    tf.keras.Input(shape=64),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(784, activation='sigmoid') # sigmoid activation function as want data to be between 0 and 1
])

# Create the Discriminator Network
# This will take in both real and fake data. That means size 784
discriminator = tf.keras.Sequential([
    tf.keras.Input(shape=784), 
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') # outputs a number between 0 and 1
])

def generateFakeData(batch_size):
    # generate batch_size length array of random noise
    # utilize generative network and store outputs
    # return outputs from generative network
    return

# Losses
batch_size = 32
fake_labels = tf.zeros(batch_size)
real_labels = tf.ones(batch_size)

def discriminator_loss(fake_preds, real_preds):
    fake_loss = tf.keras.losses.BinaryCrossentropy(fake_labels, fake_preds)
    real_loss = tf.keras.losses.BinaryCrossentropy(real_labels, real_preds)
    return fake_loss + real_loss


n_epochs = 50

for epoch in range(n_epochs):

    # sample minibatch size m of real data
    for minibatch_real in training_ds_batched: 

        # sample minibatch size m of noise samples
        minibatch_fake = generateFakeData(batch_size)


        with tf.GradientTape() as tape:
            fake_preds = discriminator(minibatch_fake)
            real_preds = discriminator(minibatch_real)
            loss = discriminator_loss(fake_preds, real_preds)

        gradient = tape.gradient(loss, discriminator.trainable_variables)

    # pair real data with label of 1. pair fake data with label of 0
    # 