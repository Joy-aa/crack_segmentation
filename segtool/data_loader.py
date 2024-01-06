import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import random
from PIL import Image
import matplotlib.pyplot as plt
import torch
import sys
sys.path.append("/home/wj/local/crack_segmentation")
from segtool import edge_utils
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps
import albumentations as A

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
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return img, mask #torch.from_numpy(np.array(mask, dtype=np.int64))
    
    # @classmethod
    # def decode_target(cls, target):
    #     target[target == 255] = 19
    #     #target = target.astype('uint8') + 1
    #     return cls.train_id_to_color[target]

    def __len__(self):
        return len(self.img_fnames)


class ImgDataSetJoint(Dataset):
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

        mname = self.mask_fnames[i]
        # mname = fname.split('.')[0] + '.png'
        mpath = os.path.join(self.mask_dir, mname)
        mask = Image.open(mpath).convert('L') 
        mask_copy = np.array(mask)
        # mask_copy[mask_copy < 127 ] = 0
        # mask_copy[mask_copy > 0 ] = 255
        mask_copy[mask_copy == 38 ] = 0
        mask_copy[mask_copy == 75 ] = 255
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
    

class CrackDataSetWithEdge(Dataset):

    def __init__(self, mode : str, path, transform= None):
        super(CrackDataSetWithEdge, self).__init__()
        self.mode = mode
        self.image_paths = []
        self.label_paths = []
        
        for dir_path in path:
            
            filenames = sorted(os.listdir(os.path.join(dir_path, 'images')))
            train_filenames, val_filenames = train_test_split(
                filenames, test_size=0.3, random_state=42)
            dataset_filenames = {
                'train': train_filenames,
                'val': val_filenames
            }
            self.image_paths += [os.path.join(dir_path, 'images', i)
                                 for i in dataset_filenames[mode]]
            self.label_paths += [os.path.join(dir_path, 'labels', i.rpartition('.')[0] + '.png')
                                 for i in dataset_filenames[mode]]
        self.transform = A.Compose(
            [A.RandomCrop(width=400, height=400,p = 1), 
            A.HorizontalFlip(p=0.5),        
            ],
            additional_targets={'label': 'image'}
        )
        self.images = []
        self.labels = []
        self.pixel_transform = A.Compose([
            A.GaussNoise(p=0.2),
            A.OneOf([
                    A.MotionBlur(p=.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                    ], p=0.2),
            A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.Sharpen(),
                    A.Emboss(),
                    A.RandomBrightnessContrast(p = 0.5),
                    ], p=0.3),
            A.HueSaturationValue(p=0.3),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, index):
        image = np.array(ImageOps.exif_transpose(Image.open(self.image_paths[index])))
        label = np.array(ImageOps.exif_transpose(Image.open(self.label_paths[index])))
        if(image.shape[0] != 400 and image.shape[1] != 400):
            paddings = A.PadIfNeeded(p=1, min_height=400, min_width=400)(image=image, mask=label)
            image, label = paddings["image"], paddings["mask"]
        if self.mode == 'train':
            augmented_result = self.transform(image=image,label=label)
            
            image = self.pixel_transform(image=augmented_result['image'])['image']
            label = augmented_result['label']
        else:
            image = A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(image=image)['image']
        
        # return torch.tensor(image).transpose(1,2).transpose(0,1), torch.tensor(label)
        img = torch.tensor(image).transpose(1,2).transpose(0,1)
        mask = torch.tensor(label).long()

        _edgemap = (mask.numpy())
        # print(_edgemap.shape)
        _edgemap = edge_utils.mask_to_onehot(_edgemap, 1)
        # print(_edgemap.shape)

        _edgemap = edge_utils.onehot_to_binary_edges(_edgemap, 1, 1)

        edgemap = torch.from_numpy(_edgemap).float()
        # print(edgemap)

        return img, mask, edgemap
    

class CrackDataset(Dataset):
    def __init__(self, mode : str, path, transform= None):
        super(CrackDataset, self).__init__()
        self.mode = mode
        self.image_paths = []
        self.label_paths = []
        
        for dir_path in path:
            
            filenames = sorted(os.listdir(os.path.join(dir_path, 'imgs')))
            # train_filenames = filenames
            train_filenames, val_filenames = train_test_split(
                filenames, test_size=0.2, random_state=42)
            dataset_filenames = {
                'train': train_filenames,
                'val': val_filenames
            }
            self.image_paths += [os.path.join(dir_path, 'imgs', i)
                                 for i in dataset_filenames[mode]]
            self.label_paths += [os.path.join(dir_path, 'masks', i.rpartition('.')[0] + '.png')
                                 for i in dataset_filenames[mode]]
        self.transform = A.Compose(
            [A.RandomCrop(width=224, height=224,p = 1), 
            A.HorizontalFlip(p=0.5),        
            ],
            additional_targets={'label': 'image'}
        )
        self.images = []
        self.labels = []
        self.pixel_transform = A.Compose([
            A.GaussNoise(p=0.2),
            A.OneOf([
                    A.MotionBlur(p=.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                    ], p=0.2),
            A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.Sharpen(),
                    A.Emboss(),
                    A.RandomBrightnessContrast(p = 0.5),
                    ], p=0.3),
            A.HueSaturationValue(p=0.3),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        
        
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, index):
        image = np.array(ImageOps.exif_transpose(Image.open(self.image_paths[index])))
        label = np.array(ImageOps.exif_transpose(Image.open(self.label_paths[index])))
        if(image.shape[0] != 224 and image.shape[1] != 224):
            paddings = A.PadIfNeeded(p=1, min_height=224, min_width=224)(image=image, mask=label)
            image, label = paddings["image"], paddings["mask"]
        if self.mode == 'train':
            augmented_result = self.transform(image=image,label=label)
            
            image = self.pixel_transform(image=augmented_result['image'])['image']
            label = augmented_result['label']
        else:
            image = A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(image=image)['image']
        
        return torch.tensor(image).transpose(1,2).transpose(0,1), torch.tensor(label)
    

if __name__=="__main__":
    train_Dataset = CrackDataSetWithEdge('val',['/mnt/hangzhou_116_homes/zek/crackseg9k/'])
    print(len(train_Dataset.image_paths))
    train_loader = torch.utils.data.DataLoader(train_Dataset, batch_size=8, shuffle=True, num_workers=8)
    for i, (input, target, edge) in enumerate(train_loader):
            input_var  = input
            target_var = target
    print(len(train_loader))
    print(len(train_Dataset))
