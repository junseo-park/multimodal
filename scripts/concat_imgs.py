import pandas as pd
import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt
import h5py
import cv2
import pickle

# Load data
h5 = h5py.File('../data/nyu_depth_v2_labeled.mat', 'r')

# Extract depths and images
depths = h5.get('depths').value.transpose(0, 2, 1)
images = h5.get('images').value.transpose(0, 3, 2, 1)

def convert_depth(gray, cm='jet'):
    """Convert an image from grayscale (1 color channel) to RGB (3 color channels)"""
    cmap = plt.get_cmap(cm)
    
    rgba_img = cmap(gray / np.max(gray))
    rgb_img = np.delete(rgba_img, 3, 2)
    
    return rgb_img

rgb_depths = [convert_depth(img) for img in depths]

# Concatenate images
concat_imgs = np.empty((images.shape[0], images.shape[1], images.shape[2] * 2, images.shape[3]))
for i in range(images.shape[0]):
    concat_imgs[i] = np.concatenate([images[i] / 255., rgb_depths[i]], axis=1)

print('Done concatenating! Beginning save...')
    
# Save as .mat
savemat('../data/concat_imgs.mat', {'data':concat_imgs})

# with open('concat_imgs.pkl', 'wb') as file:
#     pickle.dump(concat_imgs, file)

h5.close()