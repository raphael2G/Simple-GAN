import tensorflow as tf
import numpy as np
from load_data import loadMnistDataset
from checkpoint_statistics import checkpointModel

import wandb

model_version = 'v6'

# Starting wandb
wandb.init(
    # set the wandb project where this run will be logged
    project="simple-gan",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0002,
    "architecture": "GAN",
    "dataset": "MNIST",
    "epochs": 10,
    }
)


# Steps to creating and training our own GAN utilizing the MNIST Dataset

# - - - - - - Loading MNIST Dataset - - - - - - 

BATCH_SIZE = 128
# Load MNIST dataset with batch size of 64
training_ds_batched = loadMnistDataset()

# - - - - - - Loading MNIST Dataset - - - - - - 

# - - - - - - Generating Fake Data  - - - - - - 

# Generating Fake Data
NOISE_SIZE = 10

# generates noise with batch_size rows and NOISE_SIZE colomns
def generateRandomNoise(batch_size=BATCH_SIZE, noise_size=NOISE_SIZE):
    random_noise_np = np.random.normal(size=(batch_size, noise_size))
    random_noise_tensor = tf.convert_to_tensor(random_noise_np)
    return tf.reshape(random_noise_tensor, [batch_size, noise_size])

# - - - - - - Generating Fake Data  - - - - - - 

# - - - - - -   Creating Networks   - - - - - - 
# Create Generator Network 
# this will take in random noise of size (64, 1), and output generated data of size (784,1)
# Creating new generator
generator = tf.keras.Sequential([
    tf.keras.Input(shape=NOISE_SIZE),
    tf.keras.layers.Dense(256),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dense(512),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dense(1024),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dense(784, activation=tf.keras.activations.tanh) # tanh activation function as want data to be between 0 and 1
], name='Generator')

# Create the Discriminator Network
# This will take in both real and fake data. That means size 784
discriminator = tf.keras.Sequential([
    tf.keras.Input(shape=784), 
    tf.keras.layers.Dense(512),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dense(256),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dense(1, activation='sigmoid') # outputs a scalar
], name='Discriminator')

# Loading Generator from Previously Trained Example 
# generator = tf.keras.models.load_model('savedModels/generator-v3/epoch-100', compile=False)
# discriminator = tf.keras.models.load_model('savedModels/discriminator-v3/epoch-100', compile=False)
# start_epoch = 100

# - - - - - -   Creating Networks   - - - - - - 

# - - - - - - Establishing Loss Functions, Optimizer, and Metrics - - - - - - 

# Losses

# Original - Probably Incorrect
# def calculate_discriminator_loss(fake_preds, real_preds):
#     fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_preds, labels=tf.zeros(fake_preds.shape)))
#     real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_preds, labels=tf.ones(real_preds.shape)))
#     total_loss = fake_loss + real_loss
#     return total_loss

# def calculate_generator_loss(fake_preds):
#     generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_preds, labels=tf.zeros(fake_preds.shape)))
#     return generator_loss

# New, hopefully correct
# Utilizes equation in paper


# Custom Loss Function
# def calculate_discriminator_loss(fake_preds, real_preds):
#     return tf.reduce_mean(tf.math.log_sigmoid(real_preds) + tf.math.log_sigmoid(tf.ones(fake_preds.shape) - fake_preds))

# def calculate_generator_loss(fake_preds):
#     return tf.reduce_mean(-tf.math.log_sigmoid(tf.ones(fake_preds.shape) - fake_preds))


loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
def calculate_discriminator_loss(fake_preds, real_preds):
    fake_preds_loss = loss_fn(tf.zeros(fake_preds.shape), fake_preds)
    real_preds_loss = loss_fn(tf.ones(real_preds.shape), real_preds)
    return fake_preds_loss + real_preds_loss

def calculate_generator_loss(fake_preds):
    return loss_fn(tf.ones(fake_preds.shape), fake_preds)


# Create the optimizer
LEARNING_RATE = 0.0002
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)


# initialize metrics
generator_loss_metric = tf.keras.metrics.Mean()
discriminator_loss_metric = tf.keras.metrics.Mean()
real_sample_accuracy = tf.keras.metrics.BinaryAccuracy()
fake_sample_accuracy = tf.keras.metrics.BinaryAccuracy()

# - - - - - - Establishing Loss Functions - - - - - - 

# - - - - - - Generating Histories to Graph - - - - - - -

n_epochs = 200

for epoch in range(n_epochs):

    # sample minibatch size m of real data
    for step, minibatch_real in enumerate(training_ds_batched): 

        # sample minibatch size m of noise samples
        noise = generateRandomNoise(BATCH_SIZE, NOISE_SIZE)
        minibatch_fake = generator(noise)

        # calculate loss for discriminiator
        with tf.GradientTape() as tape:
            fake_preds = discriminator(minibatch_fake)
            real_preds = discriminator(minibatch_real)
            discriminator_loss = calculate_discriminator_loss(fake_preds, real_preds)

        # calculate gradient for discriminator
        discriminator_gradients = tape.gradient(discriminator_loss, discriminator.trainable_variables)
        discriminator_loss_metric.update_state(discriminator_loss)
        real_sample_accuracy.update_state(tf.ones(real_preds.shape), real_preds)
        fake_sample_accuracy.update_state(tf.zeros(fake_preds.shape), fake_preds)
        total_sample_accuracy = (real_sample_accuracy.result().numpy() + fake_sample_accuracy.result().numpy()) / 2

        # apply gradients
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

        # train generator network
        # calculate loss for geneator
        noise = generateRandomNoise(BATCH_SIZE, NOISE_SIZE)

        with tf.GradientTape() as tape:
            # sample minibatch size m of noise samples
            minibatch_fake = generator(noise)
            fake_preds = discriminator(minibatch_fake)
            generator_loss = calculate_generator_loss(fake_preds)

        # calculate gradient for generator
        generator_gradients = tape.gradient(generator_loss, generator.trainable_variables)
        generator_loss_metric.update_state(generator_loss)

        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))


    print('Epoch: %i Generator Loss: %.4f Discriminator Loss: %.4f' %(epoch, generator_loss_metric.result().numpy(), discriminator_loss_metric.result().numpy()))
    print('Real Sample Accuracy: %.4f Fake Sample Accuracy: %.4f' %(real_sample_accuracy.result().numpy(), fake_sample_accuracy.result().numpy()))
    wandb.log({
        "generator_loss": generator_loss_metric.result().numpy(), 
        "discriminator_loss": discriminator_loss_metric.result().numpy(),
        "real_sample_accuracy": real_sample_accuracy.result().numpy(), 
        "fake_sample_accuracy": fake_sample_accuracy.result().numpy(),
        "total_sample_accuracy": (total_sample_accuracy), 
    })
    generator_loss_metric.reset_states()
    discriminator_loss_metric.reset_states()
    real_sample_accuracy.reset_states()
    fake_sample_accuracy.reset_states()
    total_sample_accuracy = 0 # this is purely book keeping. not functional

    if epoch % 50 == 0 and not epoch == 0: 
        generator.save('savedModels/generator-' + model_version + '/epoch-%i' %epoch)
        discriminator.save('savedModels/discriminator-' + model_version + '/epoch-%i' %epoch)


    checkpointModel(generator, epoch, model_version, noise_size=NOISE_SIZE)

wandb.finish()
generator.save('savedModels/generator-' + model_version + '/final')
discriminator.save('savedModels/discriminator-' + model_version + '/final')

