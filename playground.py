import tensorflow as tf
import numpy as np
from matplotlib import pyplot
import tensorflow_datasets as tfds

# Steps to creating and training our own GAN utilizing the MNIST Dataset

# Load in MNIST dataset
BATCH_SIZE = 64
(train_X, train_y), (test_X, test_y) = tf.keras.datasets.mnist.load_data()
train_X = tf.convert_to_tensor(train_X)
train_X_ds = tf.data.Dataset.from_tensor_slices(train_X)
batched_training_ds = train_X_ds.batch(BATCH_SIZE)



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

NOISE_SIZE = 64

# generates noise with batch_size rows and NOISE_SIZE colomns
def generateFakeData(batch_size, NOISE_SIZE):
    random_noise_np = np.random.normal(size=(batch_size, NOISE_SIZE))
    return tf.convert_to_tensor(random_noise_np)


# Losses
batch_size = 32
fake_labels = tf.zeros(batch_size)
real_labels = tf.ones(batch_size)

def calculate_discriminator_loss(fake_preds, real_preds):
    fake_loss = tf.keras.losses.BinaryCrossentropy(fake_labels, fake_preds)
    real_loss = tf.keras.losses.BinaryCrossentropy(real_labels, real_preds)
    return fake_loss + real_loss

def calculate_generator_loss(fake_preds):
    generator_loss = tf.keras.losses.BinaryCrossentropy(fake_labels, fake_preds)
    return generator_loss

# initialize metrics
generator_loss_metric = tf.keras.metrics.Mean()
discriminator_loss_metric = tf.keras.metrics.Mean()


# Create the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

n_epochs = 50

for epoch in range(n_epochs):
    # sample minibatch size m of real data
    for step, minibatch_real in enumerate(training_ds_batched): 

        # sample minibatch size m of noise samples
        minibatch_fake = generateFakeData(batch_size, NOISE_SIZE)

        # calculate loss for discriminiator
        with tf.GradientTape() as tape:
            fake_preds = discriminator(minibatch_fake)
            real_preds = discriminator(minibatch_real)
            discriminator_loss = calculate_discriminator_loss(fake_preds, real_preds)

        # calculate gradient for discriminator
        discriminator_gradients = tape.gradient(discriminator_loss, discriminator.trainable_weights)
        discriminator_loss_metric.update_state(discriminator_loss)
        
        # apply gradients
        optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_weights))

        if step % 200 == 0:
            print("Training loss (for one batch) at step %d. Generator: %.4f" % (step, float(generator_loss)))
            print("Seen so far: %s samples" % ((step + 1) * batch_size))

    # train generator network
    # sample minibatch size m of noise samples
        minibatch_fake = generateFakeData(batch_size, NOISE_SIZE)

    # calculate loss for geneator
        with tf.GradientTape() as tape:
            fake_preds = discriminator(minibatch_fake)
            generator_loss = calculate_generator_loss(fake_preds)

        # calculate gradient for generator
        generator_gradients = tape.gradient(generator_loss, generator.trainable_weights)
        generator_loss_metric.update_state(generator_loss)

        optimizer.apply_gradients(zip(generator_gradients, generator.trainable_weights))

    print('Epoch: %i Generator Loss: %.4f Discriminator Loss: %.4f',(epoch, generator_loss_metric.result().numpy(), discriminator_loss_metric.result().numpy()))

generator.save('saved_models/generator')
discriminator.save('saved_model/discriminator')


