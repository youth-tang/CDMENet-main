import random
import os
from PIL import Image,ImageFilter,ImageDraw,ImageEnhance,ImageOps,ImageChops
import numpy as np
import h5py
from PIL import ImageStat
import cv2

def load_data(img_path):


    gt_path = img_path.replace('.jpg','.h5').replace('images','labels')
    img = Image.open(img_path).convert('RGB')
    img = img.resize((1296, 864), Image.ANTIALIAS)
    gt_file = h5py.File(gt_path)
    if ('count_value' in gt_file.keys()):
        # for unlabeled images, density maps are unavailable
        flag = 1
        target = np.asarray(gt_file['count_value'])

    else:
        flag = 0
        target = np.asarray(gt_file['density'])
        target = cv2.resize(target, (162, 108), interpolation=cv2.INTER_CUBIC) * 64

    return img, target, flag