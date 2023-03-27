import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import glob
import math
import os

base_path = 'images/v5'
complete_path = os.path.join(base_path, 'complete')
epochs = os.listdir(base_path)
gif_name = os.path.join(base_path, 'v5.gif')
duration = 200 # duration of each frame in milliseconds

try: 
    os.mkdir(complete_path)
except: 
    print('Directory already exists. Rewriting.')


for epoch_folder in epochs:
    if epoch_folder == 'complete': 
        continue
    
    # load the images
    image_paths = glob.glob(os.path.join(base_path, epoch_folder, '*.png'))
    images = [np.asarray(Image.open(path)) for path in image_paths]

    # calculate the number of rows and columns needed for a square grid
    num_images = len(images)
    num_cols = int(math.sqrt(num_images))
    num_rows = int(math.ceil(num_images / num_cols)) if not num_cols == 0 else 1

    # create a square grid of subplots
    fig, axs = plt.subplots(num_rows, num_cols)
    fig.subplots_adjust(wspace=.002, hspace=0.002)

    # display each image in a separate subplot
    for i, ax in enumerate(axs.flat):
        if i < len(images):
            ax.imshow(images[i])
        ax.set(xticks=[], yticks=[])
        ax.axis('off')
        ax.set_frame_on(False)

    # show the plot
    plt.axis('off')
    plt.savefig(os.path.join(complete_path, epoch_folder))
    plt.close()

# Get a list of image file names in the folder
images = [os.path.join(complete_path, file_name) for file_name in os.listdir(complete_path) if file_name.endswith('.png') or file_name.endswith('.jpg')]

def sort_key(folder_name):
    name = os.path.split(folder_name)[-1]
    return int(name.split('-')[1][:-4])

sorted_folders = sorted(images, key=sort_key)

# Open each image file and append them to a list
frames = []
for image in sorted_folders:
    frames.append(Image.open(image))

# Save the frames as a GIF file
frames[0].save(gif_name, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)

