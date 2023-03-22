import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os


def generateFakeData(model_reference, batch_size=64, noise_size=100):
    random_noise_np = np.random.normal(size=(batch_size, noise_size))
    random_noise_tensor = tf.convert_to_tensor(random_noise_np)
    random_noise_tensor = tf.reshape(random_noise_tensor, [batch_size, noise_size])
    fake_data = model_reference(random_noise_tensor)

    return fake_data

def checkpointModel(model_reference, epoch, dir='images/v4', batch_size=64, noise_size=100):
    path = os.path.join(dir, ('v4-' + epoch))
    os.mkdir(path)

    generated_imgs_batched = generateFakeData(model_reference, batch_size, noise_size).numpy()

    for i, vector in enumerate(generated_imgs_batched):
        formated_arr = np.reshape(vector, (28, 28, 1))
        plt.imshow(formated_arr, cmap='gray', vmin=0, vmax=1)
        plt.savefig(path + '-' + i)