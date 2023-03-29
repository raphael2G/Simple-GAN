from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from load_data import loadMnistDataset

generator1 = tf.keras.models.load_model('savedModels/generator-v3/epoch-250', compile=False)
discriminator1 = tf.keras.models.load_model('savedModels/discriminator-v3/epoch-250', compile=False)

generator2 = tf.keras.models.load_model('savedModels/generator-v3/final', compile=False)
discriminator2 = tf.keras.models.load_model('savedModels/discriminator-v3/final', compile=False)


# Generating Fake Data
NOISE_SIZE = 100

# generates noise with batch_size rows and NOISE_SIZE colomns
def generateFakeData(model_reference, batch_size, noise_size=NOISE_SIZE):
    random_noise_np = np.random.normal(size=(batch_size, noise_size))
    random_noise_tensor = tf.convert_to_tensor(random_noise_np)
    random_noise_tensor = tf.reshape(random_noise_tensor, [batch_size, noise_size])
    fake_data = model_reference(random_noise_tensor)
    return fake_data

# Test Our Model

print('- - - - - - Fake Data Predicitons - - - - - -')
random_noise_np = np.random.normal(size=(1, 100))
random_noise_tensor = tf.convert_to_tensor(random_noise_np)
random_noise_tensor = tf.reshape(random_noise_tensor, [1, 100])
fake_data_1 = generator1(random_noise_tensor)
fake_labels_1 = discriminator1(fake_data_1).numpy()
fake_data_2 = generator2(random_noise_tensor)
fake_labels_2 = discriminator2(fake_data_2).numpy()

print(fake_labels_1)
print(fake_labels_2)


np_data = fake_data_1.numpy()[0]
img = np.reshape(np_data, (28, 28, 1))
plt.imshow(img, cmap='gray', vmin=0, vmax=1)
np_data = fake_data_1.numpy()[0]
img = np.reshape(np_data, (28, 28, 1))
plt.imshow(img, cmap='gray', vmin=0, vmax=1)
plt.show()
plt.show()