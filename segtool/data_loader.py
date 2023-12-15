import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import random
from PIL import Image
import matplotlib.pyplot as plt
import torch
from segtool import edge_utils
import cv2

class ImgDataSet(Dataset):

    def __init__(self, img_dir, img_fnames, img_transform, mask_dir, mask_fnames, mask_transform):
        self.img_dir = img_dir
        self.img_fnames = img_fnames
        self.img_transform = img_transform

        self.mask_dir = mask_dir
        self.mask_fnames = mask_fnames
        self.mask_transform = mask_transform

        self.seed = np.random.randint(2147483647)

    def __getitem__(self, i):
        fname = self.img_fnames[i]
        fpath = os.path.join(self.img_dir, fname)
        img = Image.open(fpath)
        if self.img_transform is not None:
            random.seed(self.seed)
            img = self.img_transform(img)
            #print('image shape', img.shape)

        mname = self.mask_fnames[i]
        mpath = os.path.join(self.mask_dir, mname)
        mask = Image.open(mpath)
        #print('khanh1', np.min(test[:]), np.max(test[:]))
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
            #print('mask shape', mask.shape)
            #print('khanh2', np.min(test[:]), np.max(test[:]))

        return img, mask #torch.from_numpy(np.array(mask, dtype=np.int64))
    
    # @classmethod
    # def decode_target(cls, target):
    #     target[target == 255] = 19
    #     #target = target.astype('uint8') + 1
    #     return cls.train_id_to_color[target]

    def __len__(self):
        return len(self.img_fnames)


class ImgDataSetJoint(Dataset):
    def __init__(self, img_dir, img_fnames, joint_transform, mask_dir, mask_fnames, img_transform = None, mask_transform = None):
        self.joint_transform = joint_transform

        self.img_dir = img_dir
        self.img_fnames = img_fnames
        self.img_transform = img_transform

        self.mask_dir = mask_dir
        self.mask_fnames = mask_fnames
        self.mask_transform = mask_transform

        self.seed = np.random.randint(2147483647)

    def __getitem__(self, i):
        fname = self.img_fnames[i]
        fpath = os.path.join(self.img_dir, fname)
        img = Image.open(fpath)

        mname = self.mask_fnames[i]
        mpath = os.path.join(self.mask_dir, mname)
        mask = Image.open(mpath)

        if self.joint_transform is not None:
            img, mask = self.joint_transform([img, mask])

        #debug
        # img = np.asarray(img)
        # mask = np.asarray(mask)
        # plt.subplot(121)
        # plt.imshow(img)
        # plt.subplot(122)
        # plt.imshow(img)
        # plt.imshow(mask, alpha=0.4)
        # plt.show()

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return img, mask #torch.from_numpy(np.array(mask, dtype=np.int64))

    def __len__(self):
        return len(self.img_fnames)
    

class CrackDataSet(Dataset):

    def __init__(self, img_dir, img_fnames, img_transform, mask_dir, mask_fnames, mask_transform):
        self.img_dir = img_dir
        self.img_fnames = img_fnames
        self.img_transform = img_transform

        self.mask_dir = mask_dir
        self.mask_fnames = mask_fnames
        self.mask_transform = mask_transform

        self.seed = np.random.randint(2147483647)

    def __getitem__(self, i):
        fname = self.img_fnames[i]
        fpath = os.path.join(self.img_dir, fname)
        img = Image.open(fpath)
        if self.img_transform is not None:
            random.seed(self.seed)
            img = self.img_transform(img)

        # mname = self.mask_fnames[i]
        mname = fname.split('.')[0] + '.png'
        mpath = os.path.join(self.mask_dir, mname)
        mask = Image.open(mpath).convert('L') 
        mask_copy = np.array(mask)
        mask_copy[mask_copy < 127 ] = 0
        mask_copy[mask_copy > 0 ] = 255
        # mask_copy[mask_copy == 38 ] = 0
        # mask_copy[mask_copy == 75 ] = 255
        mask = Image.fromarray(mask_copy.astype(np.uint8))
        
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        
        mask = mask.squeeze(0)
        
        _edgemap = (mask.numpy())
        # print(_edgemap.shape)
        _edgemap = edge_utils.mask_to_onehot(_edgemap, 1)
        # print(_edgemap.shape)

        _edgemap = edge_utils.onehot_to_binary_edges(_edgemap, 1, 1)

        edgemap = torch.from_numpy(_edgemap).float()
        # print(edgemap)

        return img, mask, edgemap

    def __len__(self):
        return len(self.img_fnames)