import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import progressbar
import cv2

from models.psp.pspnet import PSPNet
from dataset import OfflineDataset, SplitTransformDataset
from util.image_saver import tensor_to_im, tensor_to_gray_im, tensor_to_seg
from util.hyper_para import HyperParameters
from eval_helper import process_high_res_im, process_im_single_pass

import os
from os import path
from argparse import ArgumentParser
import time

import sys
sys.path.append("/home/wj/local/crack_segmentation")
from metric import *

class Parser():
    def parse(self):
        self.default = HyperParameters()
        self.default.parse(unknown_arg_ok=True)

        parser = ArgumentParser()

        parser.add_argument('--dir', default='/home/wj/dataset/crack/image', help='Directory with testing images')
        parser.add_argument('--model',default='/home/wj/local/crack_segmentation/CascadePSP/checkpoints/model_44300', help='Pretrained model')
        parser.add_argument('--output', default='/home/wj/local/crack_segmentation/CascadePSP/results', help='Output directory')

        parser.add_argument('--global_only', help='Global step only', action='store_true')

        parser.add_argument('--L', help='Parameter L used in the paper', type=int, default=900)
        parser.add_argument('--stride', help='stride', type=int, default=450)

        parser.add_argument('--clear', help='Clear pytorch cache?', action='store_true')

        parser.add_argument('--ade', help='Test on ADE dataset?', action='store_true')

        args, _ = parser.parse_known_args()
        self.args = vars(args)

    def __getitem__(self, key):
        if key in self.args:
            return self.args[key]
        else:
            return self.default[key]

    def __str__(self):
        return str(self.args)

# Parse command line arguments
para = Parser()
para.parse()
print('Hyperparameters: ', para)

# Construct model
model = nn.DataParallel(PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50').cuda())
model.load_state_dict(torch.load(para['model']))

batch_size = 1

if para['ade']:
    val_dataset = SplitTransformDataset(para['dir'], need_name=True, perturb=False, img_suffix='_im.jpg')
else:
    # print('val')
    val_dataset = OfflineDataset(para['dir'], need_name=True, resize=False, do_crop=False)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=2)

os.makedirs(para['output'], exist_ok=True)

epoch_start_time = time.time()
model = model.eval()
metrics = {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
    }

with torch.no_grad():
    for im, seg, gt, name in progressbar.progressbar(val_loader):
        im, seg, gt = im, seg, gt

        if para['global_only']:
            if para['ade']:
                # GTs of small objects in ADE are too coarse -- less upsampling is better
                images = process_im_single_pass(model, im, seg, 224, para)
            else:
                images = process_im_single_pass(model, im, seg, para['L'], para)
        else:
            images = process_high_res_im(model, im, seg, para, name, aggre_device='cuda:1')

        images['im'] = im
        images['seg'] = seg
        images['gt'] = gt

        # Suppress close-to-zero segmentation input
        for b in range(seg.shape[0]):
            if (seg[b]+1).sum() < 2:
                images['pred_224'][b] = 0

        # Save output images
        for i in range(im.shape[0]):
            print(torch.max(images['pred_224'][i]))
            print(torch.max(gt[i]))
            metric = calc_metric(images['pred_224'][i], gt[i], mode='tensor', threshold=0.5, max_value=1)
            metrics['accuracy'] += metric['accuracy'] / len(val_loader)
            metrics['precision'] += metric['precision'] / len(val_loader)
            metrics['recall'] += metric['recall'] / len(val_loader)
            metrics['f1'] += metric['f1'] / len(val_loader)
            print(metric)
            # cv2.imwrite(path.join(para['output'], '%s_im.png' % (name[i]))
            #     ,cv2.cvtColor(tensor_to_im(im[i]), cv2.COLOR_RGB2BGR))
            # cv2.imwrite(path.join(para['output'], '%s_seg.png' % (name[i]))
            #     ,tensor_to_seg(images['seg'][i]))
            # cv2.imwrite(path.join(para['output'], '%s_gt.png' % (name[i]))
            #     ,tensor_to_gray_im(gt[i]))
            cv2.imwrite(path.join(para['output'], '%s_mask.png' % (name[i]))
                ,tensor_to_gray_im(images['pred_224'][i]))
with open('result.txt', 'a', encoding='utf-8') as fout:
            print(metrics)
            line =  "accuracy:{:.5f} | precision:{:.5f} | recall:{:.5f} | f1:{:.5f} " \
                .format(metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']) + '\n'
            fout.write(line)

print('Time taken: %.1f s' % (time.time() - epoch_start_time))