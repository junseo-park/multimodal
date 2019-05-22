import pandas as pd
import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt
import h5py
import cv2
import imageio
from central_scale_image import central_scale_images
import datetime

from pathlib import Path
DATA_DIR = Path('../data/full/')

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
    
SAMPLE_EVERY_N_FRAMES = 10
RANDOM_SAMPLE_SCALE = 1/8

def convert_depth(gray, cm='jet'):
    """Convert an image from grayscale (1 color channel) to RGB (3 color channels)"""
    cmap = plt.get_cmap(cm)
    
    rgba_img = cmap(gray / np.max(gray))
    rgb_img = np.delete(rgba_img, 3, 2)
    
    return rgb_img


##### LOG FILES #####
# Make logs directory
(DATA_DIR/'logs').mkdir(exist_ok=True)
# Initialize mismatches log file if it does not exist
mismatches = DATA_DIR/'logs'/'mismatch.log'
mismatches.touch(exist_ok=True)

seen_file = DATA_DIR/'logs'/'seen.log'
seen_file.touch(exist_ok=True)
    

##### INITIALIZE SAVE DIR #####
SAVE_DIR = DATA_DIR/'concat_imgs'
SAVE_DIR.mkdir(exist_ok=True)

##### Iterate through each subdirectory of DATA_DIR #####
subdirectories = [d for d in (DATA_DIR/'raw').iterdir() if d.is_dir()]
for subdir in subdirectories:
    
    # Identify images and depths. If not equal, save in log
    imgs = list(subdir.glob('*.ppm'))
    depths = list(subdir.glob('*.pgm'))
    
    if len(imgs) != len(depths):
        with open(mismatches, 'a') as file:
            file.write(f"Mismatch for {subdir.stem}: {len(imgs)} images and {len(depths)} depths\n")
        
    
    # Sample every nth image
    for count, i in enumerate(range(min(len(imgs), len(depths)))[::SAMPLE_EVERY_N_FRAMES]):
        # Read image and convert depth to RGB
        img, depth = imageio.imread(imgs[i]) / 255., imageio.imread(depths[i])
        rgb_depth = convert_depth(depth)
        
        # Concatenate images
        concat_img = np.concatenate([img, rgb_depth], axis=1)
        
        # Save image
        imageio.imsave(SAVE_DIR/(subdir.stem + '_' + str(count).zfill(4) + '.png'), concat_img)
        
        # Scaling; only pick RANDOM_SAMPLE_SCALE of the images to scale
        if np.random.random() <= RANDOM_SAMPLE_SCALE:
            scale_img, scale_depth = central_scale_images(img, np.expand_dims(depth, axis=2))
            rgb_scale_depth = convert_depth(np.squeeze(scale_depth, axis=2))
            
            concat_img_scale = np.concatenate([scale_img, rgb_scale_depth], axis=1)
            
            imageio.imsave(SAVE_DIR/(subdir.stem + '_' + str(count).zfill(4) + '_scale.png'), concat_img_scale)
            
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(seen_file, 'a') as file:
        file.write(subdir.stem + ': ' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '\n')