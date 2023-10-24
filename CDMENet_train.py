import sys
import os

import warnings

from model import CSRNet

from Mutual_Exc import save_checkpoint0,dice_loss,densitymap_to_densitymask,unlabel_CE_loss4v1,\
    unlabel_CE_loss3v1,unlabel_CE_loss2v1

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import torch.nn.functional as F

import numpy as np
import argparse
import json
import cv2
import random
import dataset
import time
import math


import scipy.io as io
import glob
import PIL.Image as Image
import h5py
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch CSRNet')
parser.add_argument('--train_json', default='./grape_train.json', help='path to train json')
parser.add_argument('--test_json',  default='./grape_test.json', help='path to test json')
parser.add_argument('--pre', '-p', metavar='PRETRAINED', default='', type=str, help='path to the pretrained model')
parser.add_argument('--gpu', type=str, default='0', help='GPU id to use.')
parser.add_argument('--task', type=str,  default='0', help='task id to use.')

def main():
    
    global args,best_prec1
    
    best_prec1 = 1e6
    best_prec2 = 1e6

    args = parser.parse_args()
    args.original_lr = 1e-6
    args.lr = 1e-6
    args.batch_size    = 1
    args.momentum      = 0.9
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 120
    # args.epochs = 30
    args.steps         = [-1,1,100,150]
    args.scales        = [1,1,1,1]
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 110
  

    args.channel = 10
    args.th = 0.8
    args.max_epochs = 300
    args.max_val = 0.1
    args.max_val1 = 0.1
    args.ramp_up_mult = -5
    args.k = 30
    args.n_samples = 270
    args.alpha = 0.7
    args.global_step = 0
    args.Z = 300 * ['']  ##intermediate outputs
    args.z = 300 * ['']  ##temporal outputs
    args.epsilor = 4e-2
    args.T = 0.5
    args.gap = 0.1
    args.beta=1e-3
  


    with open(args.train_json, 'r') as outfile:
        train_list = json.load(outfile)

        # 10%
        train_list_unlabel = train_list[0:11] + train_list[15:49] + train_list[53:90] + train_list[94:105] + train_list[108:150] + train_list[152:169]
        train_list_label = train_list[11:15] + train_list[49:53] + train_list[90:94] + train_list[105:108] + train_list[150:152]

    with open(args.test_json, 'r') as outfile:
        val_list = json.load(outfile)
        val_list_unlabel = []
        val_list_label = val_list

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    rand_seed = 123456
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    random.seed(rand_seed)
    
    model = CSRNet()
    model = model.cuda()

    
    criterion_mse = nn.MSELoss(size_average=False).cuda()
    criterion_cls = torch.nn.CrossEntropyLoss(size_average=False,ignore_index=10).cuda()


    optimizer = torch.optim.Adam(model.parameters(), args.lr)
                                    

    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))




    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer,epoch)

        train(train_list_unlabel, train_list_label, model, criterion_mse,criterion_cls, optimizer,epoch)

        prec1 = validate(val_list_unlabel, val_list_label, model)

        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print ('best_mae is',best_prec1)
        save_checkpoint0({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }, is_best, args.task)



def train(train_list_unlabel, train_list_label, model, criterion_mse, criterion_cls, optimizer, epoch):

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list_unlabel, train_list_label,
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.4887, 0.5294, 0.3810],
                                                     std=[0.2173, 0.1964, 0.2038]),
                            ]),
                            train=True,
                            seen=model.seen,
                            batch_size=args.batch_size,
                            num_workers=args.workers,
                            ),
        batch_size=args.batch_size)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))

    model.train()
    end = time.time()


    for i, (img, target, flag) in enumerate(train_loader):
        data_time.update(time.time() - end)

        img = Variable(img.cuda())

        target = target.type(torch.FloatTensor).unsqueeze(1).cuda()
        target = Variable(target)
        target = F.relu(target)

        density, logits2, logits3, logits4 = model(img)

        if flag.sum() == 0: 
            #processing labeled images
            target_mask_2 = densitymap_to_densitymask(target, threshold1=0.0, threshold2=0.0015)
            target_mask_3 = densitymap_to_densitymask(target, threshold1=0.0015, threshold2=0.0100)
            target_mask_4 = densitymap_to_densitymask(target, threshold1=0.0100, threshold2=1.0)

            loss_density_1 = criterion_mse(density, target)

            cos_sim2 = torch.cosine_similarity(torch.squeeze(logits2[:, 1, :, :], dim=1), target_mask_2, dim=2)
            cos_sim3 = torch.cosine_similarity(torch.squeeze(logits3[:, 1, :, :], dim=1), target_mask_3, dim=2)
            cos_sim4 = torch.cosine_similarity(torch.squeeze(logits4[:, 1, :, :], dim=1), target_mask_4, dim=2)

            cos_sim23 = torch.cosine_similarity(torch.squeeze(logits2[:, 1, :, :], dim=1), target_mask_3, dim=2)
            cos_sim24 = torch.cosine_similarity(torch.squeeze(logits2[:, 1, :, :], dim=1), target_mask_4, dim=2)
            cos_sim32 = torch.cosine_similarity(torch.squeeze(logits3[:, 1, :, :], dim=1), target_mask_2, dim=2)
            cos_sim34 = torch.cosine_similarity(torch.squeeze(logits3[:, 1, :, :], dim=1), target_mask_4, dim=2)
            cos_sim42 = torch.cosine_similarity(torch.squeeze(logits4[:, 1, :, :], dim=1), target_mask_2, dim=2)
            cos_sim43 = torch.cosine_similarity(torch.squeeze(logits4[:, 1, :, :], dim=1), target_mask_3, dim=2)

            cos_sim_max = (-1) * (cos_sim2.sum() + cos_sim3.sum()+ cos_sim4.sum())/3.0
            cos_sim_min = (1) * (cos_sim23.sum() + cos_sim24.sum() + cos_sim32.sum() + cos_sim34.sum() + cos_sim42.sum() + cos_sim43.sum()) / 6.0


            loss_cls_label2 = criterion_cls(logits2,target_mask_2)
            loss_cls_label3 = criterion_cls(logits3,target_mask_3)
            loss_cls_label4 = criterion_cls(logits4,target_mask_4)

            loss = 1 * loss_density_1 + 0.01 * (loss_cls_label2.sum()+ loss_cls_label3.sum()+ loss_cls_label4.sum())/3.0+ 1*cos_sim_min + 1*cos_sim_max

        else:
            #processing unlabeled images

            pro2u, pro3u, pro4u = F.softmax(logits2, dim=1), F.softmax(logits3, dim=1), F.softmax(logits4, dim=1)

            loss_cls_unlabel2, target_2 = unlabel_CE_loss2v1(logits2=logits2, prob3=pro3u, prob4=pro4u, th=args.th, criterion_cls=criterion_cls)
            loss_cls_unlabel3, target_3 = unlabel_CE_loss3v1(logits3=logits3, prob2=pro2u, prob4=pro4u, th=args.th, criterion_cls=criterion_cls)
            loss_cls_unlabel4, target_4 = unlabel_CE_loss4v1(logits4=logits4, prob2=pro2u, prob3=pro3u, th=args.th, criterion_cls=criterion_cls)

            cos_sim2 = torch.cosine_similarity(torch.squeeze(logits2[:, 1, :, :], dim=1), target_2, dim=2)
            cos_sim3 = torch.cosine_similarity(torch.squeeze(logits3[:, 1, :, :], dim=1), target_3, dim=2)
            cos_sim4 = torch.cosine_similarity(torch.squeeze(logits4[:, 1, :, :], dim=1), target_4, dim=2)

            cos_sim23 = torch.cosine_similarity(torch.squeeze(logits2[:, 1, :, :], dim=1), target_3, dim=2)
            cos_sim24 = torch.cosine_similarity(torch.squeeze(logits2[:, 1, :, :], dim=1), target_4, dim=2)
            cos_sim32 = torch.cosine_similarity(torch.squeeze(logits3[:, 1, :, :], dim=1), target_2, dim=2)
            cos_sim34 = torch.cosine_similarity(torch.squeeze(logits3[:, 1, :, :], dim=1), target_4, dim=2)
            cos_sim42 = torch.cosine_similarity(torch.squeeze(logits4[:, 1, :, :], dim=1), target_2, dim=2)
            cos_sim43 = torch.cosine_similarity(torch.squeeze(logits4[:, 1, :, :], dim=1), target_3, dim=2)

            cos_sim_max = (-1) * (cos_sim2.sum() + cos_sim3.sum() + cos_sim4.sum()) / 3.0
            cos_sim_min = (1) * (cos_sim23.sum() + cos_sim24.sum() + cos_sim32.sum() + cos_sim34.sum() + cos_sim42.sum() + cos_sim43.sum()) / 6.0

            loss = 0.01 * (loss_cls_unlabel2.sum() + loss_cls_unlabel3.sum() + loss_cls_unlabel4.sum()) / 3.0+ 1*cos_sim_min + 1*cos_sim_max

        losses.update(loss, img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))



def validate(val_list_unlabel, val_list_label, model):
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(val_list_unlabel, val_list_label,
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.4931, 0.5346, 0.3792],
                                                     std=[0.2217, 0.2025, 0.2085]),
                            ]), train=False
                            ),
        batch_size=args.batch_size)

    model.eval()

    mae = 0

    for i, (img, target, flag) in enumerate(test_loader):
        img = img.cuda()

        img = Variable(img)
        target = target.type(torch.FloatTensor).unsqueeze(0)
        target = target.cuda()
        target = Variable(target)  # *mask_roi_v sharp

        d1, _, _, _= model(img)
        d = d1

        # mae += abs((d.data.sum() - target.sum())) / target.sum()
        mae += abs(d.data.sum() - target.sum())

    mae_min = mae / len(test_loader)

    print(' * MAE {mae:.3f} ' .format(mae=mae_min))

    return mae_min



def adjust_learning_rate(optimizer,epoch):
    args.lr = args.original_lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr



class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
    
if __name__ == '__main__':
    main()        
