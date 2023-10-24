import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.metrics import r2_score
from matplotlib import rcParams
from scipy.ndimage.filters import gaussian_filter
import scipy
import scipy.stats
import json
import torch.nn.functional as F
from matplotlib import cm as CM
from image import *
from model999 import CSRNet
import torch
import torch.nn as nn
from torch.autograd import Variable
import cv2
from new_utils import densitymap_to_densitymask,densitymap_to_densitylevel
from torchvision import datasets, transforms


transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.4931, 0.5346, 0.3792],
                                     std=[0.2217, 0.2025, 0.2085]),])

def rsquared(pd, gt):
    """ Return R^2 where x and y are array-like."""
    pd, gt = np.array(pd), np.array(gt)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(pd, gt)
    return r_value ** 2

img_paths = []
# for path in path_sets:
#     for img_path in glob.glob(os.path.join(path, '*.jpg')):
#         img_paths.append(img_path)

with open('./grape_test.json', 'r') as outfile:
    img_paths = json.load(outfile)

model = CSRNet()
model = model.cuda()
model.eval()

checkpoint = torch.load('./10_CDMENet.tar')
print(checkpoint['epoch'])

model.load_state_dict(checkpoint['state_dict'])


mae = 0
mse=0
precount = []
gthcount = []
for i in range(len(img_paths)):
    print(img_paths[i])

    img = Image.open(img_paths[i])
    img = img.resize((1296, 864), Image.ANTIALIAS)
    img = transform(img.convert('RGB')).cuda()

    gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','labels'),'r')
    groundtruth_d = np.asarray(gt_file['density'])

    groundtruth_d = cv2.resize(groundtruth_d, (162, 108), interpolation=cv2.INTER_CUBIC) * 64

    d1,u1,u2,u3 = model(img.unsqueeze(0))

    print('et:',(d1).detach().cpu().sum().numpy())
    print('gt:',np.sum(groundtruth_d))
    precount.append((d1).detach().cpu().sum().numpy())
    gthcount.append(np.sum(groundtruth_d))

    mae += abs((d1).detach().cpu().sum().numpy()  - np.sum(groundtruth_d))
    mse += ((d1).detach().cpu().sum().numpy() - np.sum(groundtruth_d))**2
#
r2 = rsquared(precount, gthcount)
print('mae',mae/len(img_paths))
print('mse',np.sqrt(mse/len(img_paths)))
print(r2)



