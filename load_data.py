import tensorflow as tf
import numpy as np

def loadMnistDataset(batch_size=64, input_shape=(28, 28)):
    # Load in MNIST dataset as numpy array
    (train_X, train_y), (test_X, test_y) = tf.keras.datasets.mnist.load_data()

    np.random.shuffle(train_X)

    # noramlizes and flattens dataset in numpy array
    flattened = np.reshape((train_X-127.5)/127.5, [len(train_X), input_shape[0] * input_shape[1]])

    # Loads np array into a Dataset object
    train_X_ds = tf.data.Dataset.from_tensor_slices(flattened)

    # Returns the flattened, normalized, batched MNIST dataset
    return train_X_ds.batch(batch_size)
















