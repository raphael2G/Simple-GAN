import tensorflow as tf
import numpy as np
import wandb
import os
 
# SET LOCAL VARIABLES #
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
LEARNING_RATE = config.getfloat('hyperparameters', 'LEARNING_RATE')
N_EPOCHS = config.getint('hyperparameters', 'N_EPOCHS')
BATCH_SIZE = config.getint('hyperparameters', 'BATCH_SIZE') # size of minibatches fed to the model
NOISE_SIZE = config.getint('hyperparameters', 'NOISE_SIZE') # size of the seed to generate images
IMG_SHAPE = eval(config.get('hyperparameters', 'IMG_SHAPE'))
IMG_SIZE = IMG_SHAPE[0] * IMG_SHAPE[1]
GENERATOR_HIDDEN_NODES = eval(config.get('hyperparameters', 'GENERATOR_HIDDEN_NODES'))
DISCRIMINATOR_HIDDEN_NODES = eval(config.get('hyperparameters', 'DISCRIMINATOR_HIDDEN_NODES'))
MODEL_VERSION = config.get('trainingStatistics', 'MODEL_VERSION')
# SET LOCAL VARIABLES #

# - - - - - - DEFINE CHECKPOINT FUNCTION - - - - - - 
def checkpointModel(generator_reference, epoch, model_version, batch_size=5, noise_size=100):
    dir = os.path.join('images', model_version)
    
    try: 
        os.mkdir(dir)
    except: 
        print()
    
    path = os.path.join(dir, ('Epoch-' + str(epoch)))
    try: 
        os.mkdir(path)
    except: 
        print('image path already exists. Overwriting Old Data')

    random_noise_tensor = generateRandomNoise()
    generated_imgs = generator_reference(random_noise_tensor).numpy()
    formated_arr = np.reshape(generated_imgs, (batch_size, 28, 28, 1))
    with open(os.path.join(path, 'generator-' + model_version), 'x') as f:
        np.save(f, formated_arr)

# - - - - - - GENERATE RANDOM NOISE - - - - - - 
def generateRandomNoise(batch_size=BATCH_SIZE, noise_size=NOISE_SIZE):
    random_noise_np = np.random.normal(size=(batch_size, noise_size))
    random_noise_tensor = tf.convert_to_tensor(random_noise_np)

    return tf.reshape(random_noise_tensor, [batch_size, noise_size])

# - - - - - - DEFINE LOSS FUNCTIONS - - - - - - 
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
def calculate_discriminator_loss(fake_preds, real_preds):
    fake_preds_loss = loss_fn(tf.zeros(fake_preds.shape), fake_preds)
    real_preds_loss = loss_fn(tf.ones(real_preds.shape), real_preds)
    return fake_preds_loss + real_preds_loss

def calculate_generator_loss(fake_preds):
    return loss_fn(tf.ones(fake_preds.shape), fake_preds)

# - - - - - - DEFINE TRAINING LOOP - - - - - - 
def trainingLoop(generator_reference, discriminator_reference, training_ds_batched):
    # Create the optimizer
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # initialize metrics
    generator_loss_metric = tf.keras.metrics.Mean()
    discriminator_loss_metric = tf.keras.metrics.Mean()
    
    real_sample_accuracy = tf.keras.metrics.BinaryAccuracy()
    fake_sample_accuracy = tf.keras.metrics.BinaryAccuracy()

    false_negatives = tf.keras.metrics.FalseNegatives()
    false_positives = tf.keras.metrics.FalsePositives()
    true_negatives = tf.keras.metrics.TrueNegatives()
    true_positives = tf.keras.metrics.TruePositives()

    for epoch in range(N_EPOCHS):

        # sample minibatch size m of real data
        for minibatch_real in training_ds_batched: 

            # sample minibatch size m of noise samples
            noise = generateRandomNoise(BATCH_SIZE, NOISE_SIZE)
            minibatch_fake = generator_reference(noise)

            # calculate loss for discriminiator
            with tf.GradientTape() as tape:
                fake_preds = discriminator_reference(minibatch_fake)
                real_preds = discriminator_reference(minibatch_real)
                discriminator_loss = calculate_discriminator_loss(fake_preds, real_preds)

            # calculate gradient for discriminator
            discriminator_gradients = tape.gradient(discriminator_loss, discriminator_reference.trainable_variables)
            discriminator_loss_metric.update_state(discriminator_loss)

            real_sample_accuracy.update_state(tf.ones(real_preds.shape), real_preds)
            fake_sample_accuracy.update_state(tf.zeros(fake_preds.shape), fake_preds)

            # update real data
            false_negatives.update_state(tf.ones(real_preds.shape), real_preds)
            false_positives.update_state(tf.ones(real_preds.shape), real_preds)
            true_negatives.update_state(tf.ones(real_preds.shape), real_preds)
            true_positives.update_state(tf.ones(real_preds.shape), real_preds)


            # update fake data
            false_negatives.update_state(tf.zeros(fake_preds.shape), fake_preds)
            false_positives.update_state(tf.zeros(fake_preds.shape), fake_preds)
            true_negatives.update_state(tf.zeros(fake_preds.shape), fake_preds)
            true_positives.update_state(tf.zeros(fake_preds.shape), fake_preds)

            # apply gradients
            discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator_reference.trainable_variables))

            # train generator network
            # calculate loss for geneator
            noise = generateRandomNoise(BATCH_SIZE, NOISE_SIZE)

            with tf.GradientTape() as tape:
                # sample minibatch size m of noise samples
                minibatch_fake = generator_reference(noise)
                fake_preds = discriminator_reference(minibatch_fake)
                generator_loss = calculate_generator_loss(fake_preds)

            # calculate gradient for generator
            generator_gradients = tape.gradient(generator_loss, generator_reference.trainable_variables)
            generator_loss_metric.update_state(generator_loss)

            generator_optimizer.apply_gradients(zip(generator_gradients, generator_reference.trainable_variables))


        print('Epoch: %i Generator Loss: %.4f Discriminator Loss: %.4f' %(epoch, generator_loss_metric.result().numpy(), discriminator_loss_metric.result().numpy()))
        print('Real Sample Accuracy: %.4f Fake Sample Accuracy: %.4f' %(real_sample_accuracy.result().numpy(), fake_sample_accuracy.result().numpy()))
        wandb.log({
            "generator_loss": generator_loss_metric.result().numpy(), 
            "discriminator_loss": discriminator_loss_metric.result().numpy(),

            "real_sample_accuracy": real_sample_accuracy.result().numpy(), 
            "fake_sample_accuracy": fake_sample_accuracy.result().numpy(),

            "false_positives": false_positives.result().numpy(),
            "false_negatives": false_negatives.result().numpy(),
            "true_positives": true_positives.result().numpy(),
            "true_negatives": true_negatives.result().numpy(),
        })

        generator_loss_metric.reset_states()
        discriminator_loss_metric.reset_states()
        real_sample_accuracy.reset_states()
        fake_sample_accuracy.reset_states()

        if epoch % 50 == 0 and not epoch == 0: 
            generator_reference.save('savedModels/generator-' + MODEL_VERSION + '/epoch-%i' %epoch)
            discriminator_reference.save('savedModels/discriminator-' + MODEL_VERSION + '/epoch-%i' %epoch)

        checkpointModel(generator_reference, epoch, MODEL_VERSION, noise_size=NOISE_SIZE)

    # wandb.finish()
    # generator_reference.save('savedModels/generator-' + MODEL_VERSION + '/final')
    # discriminator_reference.save('savedModels/discriminator-' + MODEL_VERSION + '/final')
