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


# Define Checkpoint Function
def checkpointModel(model_reference, epoch, model_version, batch_size=5, noise_size=100):
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


    generated_imgs_batched = generateFakeData(model_reference, batch_size, noise_size).numpy()
    formated_arr = np.reshape(generated_imgs_batched, (batch_size, 28, 28, 1))
    with open(os.path.join(path, 'generator-' + model_version), 'x') as f:
        np.save(f, formated_arr)

    # how to view the data
    # for i, vector in enumerate(generated_imgs_batched):
    #     formated_arr = np.reshape(vector, (28, 28, 1))
    #     plt.imshow(formated_arr, cmap='gray', vmin=-1, vmax=1)
    #     plt.savefig(os.path.join(path, 'generator-' + 'v5' + '-' + str(i)))