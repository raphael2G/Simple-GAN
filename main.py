import tensorflow as tf
import wandb
from training_loop import trainingLoop
from load_data import loadMnistDataset

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


# Starting wandb
wandb.init(
    # set the wandb project where this run will be logged
    project="simple-gan",
    
    # track hyperparameters and run metadata
    config={
    "Architecture": "GAN",
    "Dataset": "MNIST",
    "LEARNING_RATE": LEARNING_RATE,
    "N_EPOCHS": N_EPOCHS,
    "BATCH_SIZE": BATCH_SIZE, 
    "NOISE_SIZE": NOISE_SIZE, 
    "IMG_SHAPE": IMG_SHAPE, 
    "IMG_SIZE": IMG_SIZE, 
    "GENERATOR_HIDDEN_NODES": GENERATOR_HIDDEN_NODES, 
    "DISCRIMINATOR_HIDDEN_NODES": DISCRIMINATOR_HIDDEN_NODES, 
    "MODEL_VERSION": MODEL_VERSION, 
    }
)
# - - - - - CREATE NETWORKS - - - - - 

# Create Generator Network - this will take in random noise of size (64, 1), and output generated data of size (784,1)
generator = tf.keras.Sequential([
    tf.keras.Input(shape=NOISE_SIZE),
    tf.keras.layers.Dense(GENERATOR_HIDDEN_NODES[0]),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dense(GENERATOR_HIDDEN_NODES[1]),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dense(GENERATOR_HIDDEN_NODES[2]),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dense(IMG_SIZE, activation=tf.keras.activations.tanh) # tanh activation function maps  data between -1 and 1. sigmoid did not work for some reason. if this changes, must change how data is processed (ie range between 0,1 or -1,1)
], name='Generator')

# Create the Discriminator Network - this will take in both real and fake data (size 784,1) and output a confidence [0,1]
discriminator = tf.keras.Sequential([
    tf.keras.Input(shape=IMG_SIZE), 
    tf.keras.layers.Dense(DISCRIMINATOR_HIDDEN_NODES[0]),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dense(DISCRIMINATOR_HIDDEN_NODES[1]),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dense(1, activation='sigmoid') # outputs a scalar
], name='Discriminator')
# - - - - - -   Creating Networks   - - - - - - 

batched_training_ds = loadMnistDataset()
trainingLoop(generator, discriminator, batched_training_ds)

wandb.finish()
generator.save('savedModels/generator-' + MODEL_VERSION + '/final')
discriminator.save('savedModels/discriminator-' + MODEL_VERSION + '/final')