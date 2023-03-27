# Simple-GAN
This is a simple implementation of a GAN trained in tensorflow utilizing the MNIST dataset. 

![Alt Text](https://github.com/raphael2G/Simple-GAN/blob/main/output.gif)



# What it does
This model is capable of generating images of handwritten digits. See examples below

# How it works
GANs (Generative Adversarial Networks) are a type of neural network used for unsupervised learning. It consists of two networks, a generator and a discriminator, which compete against each other in a zero-sum game. The generator attempts to create realistic data, while the discriminator tries to identify the generated data from the real data. The two networks learn from each other in an iterative process, as the generator attempts to fool the discriminator and the discriminator becomes better at recognizing generated data. This process eventually converges to a Nash equilibrium, where the generator produces realistic data that the discriminator is unable to distinguish from the real data. In essence, GANs are a way of teaching AI agents to create things without explicit instructions.

# How it was made
To create a GAN using the MNIST dataset, we first need to define a generator network and a discriminator network. The generator network is typically a deep convolutional neural network (CNN) which takes a random noise vector as input and outputs a generated image. However, for a simple proof-of-concept the generator was simply a 3-layered MLP. The discriminator network is also a 3-layered MLP which takes the generated image as input and outputs a probability score. The two networks are trained together, with the generator attempting to generate images that the discriminator is unable to distinguish from the real images. The generator and discriminator are optimized using a loss function, such as the binary cross-entropy loss for a binary classification problem. This was constructed utilizing the tensorflow API. 

# Training Data
As training is currently ongoing, there is no training data present. Current models posted in the respository have not yet reached nash equilibrium. 
