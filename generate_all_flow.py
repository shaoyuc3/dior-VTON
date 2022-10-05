"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.test_options import TestOptions
from datasets import create_dataset
from models.networks.flowStyle.networks import load_checkpoint
from models.networks.flowStyle.afwm import AFWM
import os, torch, shutil
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.util import tensor2im
from tensorboardX import SummaryWriter
import cv2, imageio
import datetime

opt = TestOptions().parse()

opt.crop_size = tuple(map(int, opt.crop_size.split(', ')))
if opt.square: #opt.crop_size >= 250:
    opt.crop_size = (opt.crop_size[0], opt.crop_size[0])
opt.square = False
print("crop_size:", opt.crop_size)

opt.isTrain = False

dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
dataset_size = len(dataset)    # get the number of images in the dataset.
print('The number of training images = %d' % dataset_size)

warp_model = AFWM(opt, 45)
    
#print(warp_model)
warp_model.eval()
warp_model.cuda()
load_checkpoint(warp_model, opt.flownet_path)

generate_out_dir = os.path.join(opt.eval_output_dir + "_%s"%opt.epoch)
print("generate images at %s" % generate_out_dir)
os.mkdir(generate_out_dir)

count = 1
for i, data in tqdm(enumerate(dataset), "generating for test split"):  # inner loop within one epoch
    with torch.no_grad():
        from_img, from_kpt, from_parse, to_img, to_kpt, to_parse, to_dense, index = data
        
        from_parse = torch.unsqueeze(from_parse, 1)
        to_parse = torch.unsqueeze(to_parse, 1)
        to_dense = torch.unsqueeze(to_dense, 1)
        
        pre_clothes_edge = torch.FloatTensor((from_parse.detach().numpy() == 5).astype(np.int))
        clothes = from_img * pre_clothes_edge
        person_clothes_edge = torch.FloatTensor((to_parse.cpu().numpy()==5).astype(np.int))
        person_clothes = to_img * person_clothes_edge
        size = to_parse.size()
        oneHot_size1 = (size[0], 25, size[2], size[3])
        densepose = torch.cuda.FloatTensor(torch.Size(oneHot_size1)).zero_()
        densepose = densepose.scatter_(1, to_dense.data.long().cuda(), 1.0)
        densepose_fore = to_dense/24.0
        face_mask = torch.FloatTensor((to_parse.cpu().numpy()==2).astype(np.int)) + torch.FloatTensor((to_parse.cpu().numpy()==4).astype(np.int))
        other_clothes_mask = torch.FloatTensor((to_parse.cpu().numpy()==1).astype(np.int)) + torch.FloatTensor((to_parse.cpu().numpy()==3).astype(np.int)) + torch.FloatTensor((to_parse.cpu().numpy()==7).astype(np.int))
        preserve_mask = torch.cat([face_mask, other_clothes_mask],1)
        concat = torch.cat([preserve_mask.cuda(), densepose, to_kpt.cuda()], 1)
        
        flow_out = warp_model(concat.cuda(), clothes.cuda(), pre_clothes_edge.cuda())

        warped_cloth, last_flow, _1, _2, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all = flow_out
        
        clothes = F.interpolate(clothes.cuda(), opt.crop_size)
        person_clothes = F.interpolate(person_clothes.cuda(), opt.crop_size)
        warped_cloth = F.interpolate(warped_cloth, opt.crop_size)
        
        rets = torch.cat([clothes, person_clothes, warped_cloth], 3)
        for i, ret in enumerate(rets):
            img = tensor2im(ret)
            imageio.imwrite(os.path.join(generate_out_dir, "generated_%d.jpg" % count), img)
            
        count += 1