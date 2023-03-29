import configparser

config = configparser.ConfigParser()
config.read('config.ini')

# Set local variables
LEARNING_RATE = config.getfloat('hyperparameters', 'LEARNING_RATE')
N_EPOCHS = config.getint('hyperparameters', 'N_EPOCHS')
BATCH_SIZE = config.getint('hyperparameters', 'BATCH_SIZE') # size of minibatches fed to the model
NOISE_SIZE = config.getint('hyperparameters', 'NOISE_SIZE') # size of the seed to generate images
IMG_SHAPE = eval(config.get('hyperparameters', 'IMG_SHAPE'))
IMG_SIZE = IMG_SHAPE[0] * IMG_SHAPE[1]
GENERATOR_HIDDEN_NODES = eval(config.get('hyperparameters', 'GENERATOR_HIDDEN_NODES'))
DISCRIMINATOR_HIDDEN_NODES = eval(config.get('hyperparameters', 'DISCRIMINATOR_HIDDEN_NODES'))

MODEL_VERSION = config.get('trainingStatistics', 'MODEL_VERSION')



