import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import random
from PIL import Image
import matplotlib.pyplot as plt
import datasets.edge_utils as edge_utils
import torch 

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
        # img = img.resize((768, 768))
        if self.img_transform is not None:
            random.seed(self.seed)
            img = self.img_transform(img)

        mname = self.mask_fnames[i]
        mpath = os.path.join(self.mask_dir, mname)
        mask = Image.open(mpath)
        # mask = mask.resize((768, 768))
        # print(mask)
        # mask_copy = np.array(mask)
        # mask_copy[mask_copy == 255] = 1
        # mask = Image.fromarray(mask_copy.astype(np.uint8))
        
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        
        _edgemap = (mask.numpy()).squeeze(0)
        # print(_edgemap.shape)
        _edgemap = edge_utils.mask_to_onehot(_edgemap, 1)
        # print(_edgemap.shape)

        _edgemap = edge_utils.onehot_to_binary_edges(_edgemap, 2, 1)

        edgemap = torch.from_numpy(_edgemap).float()
        # print(edgemap)

        return img, mask, edgemap

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