import torch.utils.data as data
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms
import torch
import copy, os, collections
import json
from .human_parse_labels import get_label_map, DF_LABEL, YF_LABEL
import pandas as pd
from utils import pose_utils
import random

def load_pose_from_json(ani_pose_dir, target_size=(256,176), orig_size=(256,176)):
    with open(ani_pose_dir, 'r') as f:
        anno = json.load(f)
    if len(anno['people']) < 1:
        a,b = target_size
        return torch.zeros((18,a,b))
    anno = list(anno['people'][0]['pose_keypoints'])
    x = np.array(anno[1::3])
    y = np.array(anno[::3])
    
    #x[8:-1] = x[9:]
    #y = np.array(anno[::3])
    #y[8:-1] = y[9:]
    x[x==0] = -1
    y[y==0] = -1
    coord = np.concatenate([x[:,None], y[:,None]], -1)
    pose  = pose_utils.cords_to_map(coord, target_size, orig_size)
    pose = np.transpose(pose,(2, 0, 1))
    pose = torch.Tensor(pose)
    return pose[:18]

class VITONPairDataset(data.Dataset):
    def __init__(self, dataroot, dim=(256,256), isTrain=True, n_human_part=8, viton=False, coord=False, aug='none'):
        super(VITONPairDataset, self).__init__()
        self.split = 'train' if isTrain else 'test'
        self.dataroot = dataroot
        self.n_human_part = n_human_part
        self.dim = dim
        self._init(viton)

    def _init(self, viton):
        self.image_dir = '%s/%s/%s' % (self.dataroot, self.split, 'image')
        self.mask_dir = '%s/%s/%s' % (self.dataroot, self.split, 'image-parse')
        self.kpt_dir = '%s/%s/%s' % (self.dataroot, self.split, 'pose')
        self.dense_dir = '%s/%s/%s' % (self.dataroot, self.split, 'densepose')
        self.cloth_dir = '%s/%s/%s' % (self.dataroot, self.split, 'cloth')
        self.cloth_mask_dir = '%s/%s/%s' % (self.dataroot, self.split, 'cloth-mask')
        with open('%s/%s_pairs.txt' % (self.dataroot, self.split), 'r') as f:
            pairs = f.readlines()
        self.name_pairs = [a.split() for a in pairs]
        self.aiyu2atr, self.atr2aiyu = get_label_map(self.n_human_part, 'lip')

        self.load_size = self.dim
        self.crop_size = self.load_size
    
        # transforms
        self.resize = transforms.Resize(self.crop_size)
        self.toTensor = transforms.ToTensor()
        self.toPIL = transforms.ToPILImage()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
    def __len__(self):
        return len(self.name_pairs)
    
    def _load_img(self, fn):
        img = Image.open(fn).convert("RGB")
        W,H = img.size
        img = self.resize(img)
        img = self.toTensor(img)
        img = self.normalize(img)
        return img, (H,W)
    
    def _load_mask(self, fn): 
        try:
            mask = Image.open(fn + ".jpg")
        except:
            mask = Image.open(fn + ".png")
        mask = self.resize(mask)
        mask = torch.from_numpy(np.array(mask))
        texture_mask = copy.deepcopy(mask)
        for atr in self.atr2aiyu:
            aiyu = self.atr2aiyu[atr]
            texture_mask[mask == atr] = aiyu
        return texture_mask
    
    def _load_kpt(self, name, orig_size=0):
        if len(orig_size) == 1:
            return load_pose_from_json(name + '_keypoints.json', self.dim)
        else:
            return load_pose_from_json(name + '_keypoints.json', self.dim, orig_size)
        
    def _load_dense(self, fn):
        dense_mask = np.load(fn + ".npy").astype(np.float32)
        #dense_mask = np.resize(dense_mask, self.crop_size)
        size = (self.crop_size[1], self.crop_size[0])
        dense_mask = cv2.resize(dense_mask, size, cv2.INTER_NEAREST)
        dense_mask = torch.from_numpy(dense_mask)
        
        return dense_mask
   
    def get_to_item(self, key):
        img,size = self._load_img(os.path.join(self.image_dir,key))
        kpt = self._load_kpt(os.path.join(self.kpt_dir,key[:-4]), size)     
        parse = self._load_mask(os.path.join(self.mask_dir, key[:-4]))
        dense = self._load_dense(os.path.join(self.dense_dir, key[:-4]))
        return img, kpt, parse.float(), dense

    def get_garment_item(self, key):
        img,_ = self._load_img(os.path.join(self.cloth_dir,key))
        _,H,W = img.size()
        kpt = torch.zeros((18, H, W)) 
        parse = self._load_mask(os.path.join(self.cloth_mask_dir, key[:-4]))
        parse[parse!=0] = 5
        return img, kpt, parse.float()
    
    def __getitem__(self, index):
        to_key, from_key = self.name_pairs[index]
        from_img, from_kpt, from_parse = self.get_garment_item(from_key)
        to_img, to_kpt, to_parse, to_dense = self.get_to_item(to_key)
    
        return from_img, from_kpt.float(), from_parse, to_img, to_kpt.float(), to_parse, to_dense, index #torch.Tensor([0])
        

class VITONVisualDataset(VITONPairDataset):
    def __init__(self, dataroot, dim=(256, 176), texture_dir="",isTrain=False, n_human_part=8):
        VITONPairDataset.__init__(self, dataroot, dim, isTrain, n_human_part=n_human_part)
        with open('%s/%s_pairs.txt' % (self.dataroot, self.split), 'r') as f:
            pairs = f.readlines()
        self.name_pairs = [a.split() for a in pairs]
        #print(self.name_pairs)
        
    def get_attr_visual_input(self):
        indx = random.randint(0, len(self.name_pairs)-1)
        to_key, from_key = self.name_pairs[indx]
        from_img, from_kpt, from_parse = self.get_garment_item(from_key)
        to_img, to_kpt, to_parse, to_dense = self.get_to_item(to_key)
        
        return from_img[None, :, :, :], from_kpt[None, :, :, :].float(), from_parse[None, :, :], to_img[None, :, :, :], to_kpt[None, :, :, :].float(), to_parse[None, :, :], to_dense[None, :, :], torch.Tensor([indx]).to(torch.int64)
    
    def get_inputs_by_key(self, from_key, to_key):
        from_img, from_kpt, from_parse = self.get_garment_item(from_key)
        to_img, to_kpt, to_parse, to_dense = self.get_to_item(to_key)
        return from_img, from_parse, to_img, to_kpt.float(), to_parse, to_dense
