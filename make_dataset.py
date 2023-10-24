import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import json
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch
import cv2



img_paths = []

with open('./grape_train.json', 'r') as outfile:
     img_paths = json.load(outfile)
     print(img_paths)

img_target=[]
num_label=0
num_unlabel=0
for img_path in img_paths:
    index = img_paths.index(img_path)
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','labels'))
    img= plt.imread(img_path)

    if img.shape[0] > 864:
        img = cv2.resize(img, (1296, 864))
        k = np.zeros((img.shape[0],img.shape[1]))
        gt = mat["image_info"][0][0][0] * 0.25
    else:
        k = np.zeros((img.shape[0], img.shape[1]))
        gt = mat["image_info"][0][0][0]

    gt_value = gt.shape[0]

    #for validation
    # if(index<0):

    #for training
    # 10%:
    if ((index <= 10) or (index >= 15 and index <= 48) or (index >= 53 and index <= 89) or (index >= 94 and index <= 104) or (index >= 108 and index <= 149) or (index >= 152 and index <= 168)):

        print('generate unlabeled image without density map supervision')
        num_unlabel = num_unlabel +1
        with h5py.File(img_path.replace('.jpg','.h5').replace('images','labels'), 'w') as hf:
             hf['count_value'] = gt_value
    else:
        num_label = num_label +1
        print('generate with map')
        for i in range(0, len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                k[int(gt[i][1]), int(gt[i][0])] = 1

        d = gaussian_filter(k,15,truncate=4)
        with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'labels'), 'w') as hf:

             hf['density'] = d

print(img_target)
print(len(img_target))
print('num_label',num_label)
print('num_unlabel',num_unlabel)



