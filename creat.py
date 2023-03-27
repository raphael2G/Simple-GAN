from PIL import Image
import os
import glob

# Set the path to the folder containing images
image_folder = 'images/v5/complete'

# Define the name and duration of the output GIF file
gif_name = 'output.gif'
duration = 100 # duration of each frame in milliseconds

# Get a list of image file names in the folder
images = [os.path.join(image_folder, file_name) for file_name in os.listdir(image_folder) if file_name.endswith('.png') or file_name.endswith('.jpg')]

def sort_key(folder_name):
    name = os.path.split(folder_name)[-1]
    return int(name.split('-')[1][:-4])

sorted_folders = sorted(images, key=sort_key)
print(sorted_folders)

# Open each image file and append them to a list
frames = []
for image in sorted_folders:
    frames.append(Image.open(image))

# Save the frames as a GIF file
frames[0].save(gif_name, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)
