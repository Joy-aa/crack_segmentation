import os
from os import path
import warnings

from torch.utils.data.dataset import Dataset
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import random
from dataset.reseed import reseed
import util.boundary_modification as boundary_modification

seg_normalization = transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            )

class OnlineTransformDataset(Dataset):
    """
    Method 0 - FSS style (class/1.jpg class/1.png)
    Method 1 - Others style (XXX.jpg XXX.png)
    """
    def __init__(self, root, need_name=False, method=0, perturb=True):
        self.root = root
        self.need_name = need_name
        self.method = method

        if method == 0:
            # Get images
            self.im_list = []
            classes = os.listdir(self.root)
            for c in classes:
                imgs = os.listdir(path.join(root, c))
                jpg_list = [im for im in imgs if 'jpg' in im[-3:].lower()]
                unmatched = any([im.replace('.jpg', '.png') not in imgs for im in jpg_list])

                if unmatched:
                    print('Number of image/gt unmatch in class ', c)
                    print('The whole class is ignored', len(jpg_list))

                    warnings.warn('Dataset unmatch error')
                else:
                    joint_list = [path.join(root, c, im) for im in jpg_list]
                    self.im_list.extend(joint_list)

        elif method == 1:
            self.im_list = [path.join(self.root, im) for im in os.listdir(self.root) if '.jpg' in im]

        print('%d images found' % len(self.im_list))

        if perturb:
            # Make up some transforms
            self.bilinear_dual_transform = transforms.Compose([
                transforms.RandomCrop((224, 224), pad_if_needed=True),
                transforms.RandomHorizontalFlip(),
            ])

            self.bilinear_dual_transform_im = transforms.Compose([
                transforms.RandomCrop((224, 224), pad_if_needed=True),
                transforms.RandomHorizontalFlip(),
            ])

            self.im_transform = transforms.Compose([
                transforms.ColorJitter(0.2, 0.05, 0.05, 0),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            # Make up some transforms
            self.bilinear_dual_transform = transforms.Compose([
                transforms.Resize(224, interpolation=Image.NEAREST), 
                transforms.CenterCrop(224),
            ])

            self.bilinear_dual_transform_im = transforms.Compose([
                transforms.Resize(224, interpolation=Image.BILINEAR), 
                transforms.CenterCrop(224),
            ])

            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

        self.gt_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.seg_transform = transforms.Compose([
            transforms.ToTensor(),
            seg_normalization,
        ])

    def __getitem__(self, idx):
        im = Image.open(self.im_list[idx]).convert('RGB')

        if self.method == 0:
            gt = Image.open(self.im_list[idx][:-3]+'png').convert('L')
        else:
            gt = Image.open(self.im_list[idx].replace('.jpg','.png')).convert('L')

        seed = np.random.randint(2147483647)

        reseed(seed)
        im = self.bilinear_dual_transform_im(im)

        reseed(seed)
        gt = self.bilinear_dual_transform(gt)

        iou_max = 1.0
        iou_min = 0.8
        iou_target = np.random.rand()*(iou_max-iou_min) + iou_min
        seg = boundary_modification.modify_boundary((np.array(gt)>0.5).astype('uint8')*255, iou_target=iou_target)

        im = self.im_transform(im)
        gt = self.gt_transform(gt)
        seg = self.seg_transform(seg)

        if self.need_name:
            return im, seg, gt, os.path.basename(self.im_list[idx][:-4])
        else:
            return im, seg, gt

    def __len__(self):
        return len(self.im_list)

if __name__ == '__main__':
    o = OnlineTransformDataset('data/train')
    o = OnlineTransformDataset('data/val')
