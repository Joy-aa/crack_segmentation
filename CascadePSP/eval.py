import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import progressbar
import cv2

from models.psp.pspnet import PSPNet
from dataset import OfflineDataset, SplitTransformDataset
from util.image_saver import tensor_to_im, tensor_to_gray_im, tensor_to_seg
from util.hyper_para import HyperParameters
from eval_helper import process_high_res_im, process_im_single_pass, safe_forward

import os
from os import path
from pathlib import Path
from argparse import ArgumentParser
import time

import sys
sys.path.append("/home/wj/local/crack_segmentation")
from metric import *
<<<<<<< HEAD
<<<<<<< HEAD
input_size=[416, 416]
=======
input_size=[160,160]
>>>>>>> 9b69e40b4731cb2975426104432bd00ad610f8d9
=======
input_size=[160,160]
>>>>>>> ac79be0045d14f416be9078b8b075926d9a31b88

class Parser():
    def parse(self):
        self.default = HyperParameters()
        self.default.parse(unknown_arg_ok=True)

        parser = ArgumentParser()

<<<<<<< HEAD
<<<<<<< HEAD
        parser.add_argument('--dir', default='/mnt/nfs/wj/data/image', help='Directory with testing images')
        parser.add_argument('--seg', default='/mnt/hangzhou_116_homes/wj/result/0825/unet1/result_img448', help='Directory with testing images')
        parser.add_argument('--model',default='/home/wj/local/crack_segmentation/CascadePSP/weights/1_2023-08-24_07:23:11/model_300690', help='Pretrained model')
=======
        parser.add_argument('--dir', default='/nfs/wj/data', help='Directory with testing images')
        parser.add_argument('--model',default='/home/wj/local/crack_segmentation/CascadePSP/weights/1_2023-08-04_00:07:51/model_250390', help='Pretrained model')
>>>>>>> 9b69e40b4731cb2975426104432bd00ad610f8d9
=======
        parser.add_argument('--dir', default='/nfs/wj/data', help='Directory with testing images')
        parser.add_argument('--model',default='/home/wj/local/crack_segmentation/CascadePSP/weights/1_2023-08-04_00:07:51/model_250390', help='Pretrained model')
>>>>>>> ac79be0045d14f416be9078b8b075926d9a31b88
        parser.add_argument('--output', default='/home/wj/local/crack_segmentation/CascadePSP/results', help='Output directory')

        # parser.add_argument('--global_only', help='Global step only', action='store_true')

        # parser.add_argument('--L', help='Parameter L used in the paper', type=int, default=256)
        # parser.add_argument('--stride', help='stride', type=int, default=64)

        # parser.add_argument('--clear', help='Clear pytorch cache?', action='store_true')

        parser.add_argument('--type', type=str, default='out', choices=['metric', 'out'])

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
epoch_start_time = time.time()
model = model.eval()

if  para['type'] == 'out':
        DIR_IMG = os.path.join(para['dir'], 'image')
        DIR_PRED = para['seg']
        DIR_GT = ''
elif para['type']  == 'metric':
        DIR_IMG = os.path.join(para['dir'], 'image')
        DIR_PRED = para['seg']
        DIR_GT = os.path.join(para['dir'], 'new_label')
else:
        print('undefind test pattern')
        exit()

os.makedirs(para['output'], exist_ok=True)
im_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
seg_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            ),
        ])
cof = 1
w, h = int(cof * input_size[0]), int(cof * input_size[1])
offset = 32
paths = [path for path in Path(DIR_IMG).glob('*.*')]
metrics = []
print('total images:{}'.format(len(paths)))
for path in paths:
    pred_list=[]
    gt_list = []
    print(str(path))
    img_0 = cv2.imread(str(path), 1)
    img_0 = np.asarray(img_0)
    if len(img_0.shape) != 3:
            print(f'incorrect image shape: {path.name}{img_0.shape}')
            continue
    img_0 = img_0[:,:,:3]

    img_height, img_width, *img_channels = img_0.shape
    # print(img_0.shape)
    # img_0 = np.reshape(img_0, (1, img_channels, img_height, img_width))
    img_1 = np.zeros((img_height, img_width)) # 生成结果

    if DIR_GT != '':
            mask_path = os.path.join(DIR_GT, path.stem+'.png')
            lab = cv2.imread(mask_path, 0)
            lab = np.reshape(lab, (img_height, img_width))
    else:
            lab = np.zeros((img_height, img_width))

    if DIR_PRED != '':
            seg_path = os.path.join(DIR_PRED, path.stem+'.png')
            seg = cv2.imread(seg_path, 0)
    else:
            seg = np.zeros((img_height, img_width))
    img_0 = im_transform(img_0).unsqueeze(0).cuda()
    # print(img_0.shape)
    seg = seg_transform(seg).unsqueeze(0).cuda()
    # print(seg.shape)
    with torch.no_grad():
        for i in range(0, img_height+h, h):
                    for j in range(0, img_width+w, w):
                        i1 = i
                        j1 = j
                        i2 = i + h
                        j2 = j + w
                        if i2>img_height:
                            i1 = max(0, img_height - h)
                            i2 = img_height
                        if j2>img_width:
                            j1 = max(0, img_width - w)
                            j2 = img_width
                        img_pat = img_0[:,:, i1:i2 + offset, j1:j2 + offset]
                        mask_pat = lab[i1:i2 + offset, j1:j2 + offset]
                        seg_pat = seg[:,:, i1:i2 + offset, j1:j2 + offset]
                        ori_shape = mask_pat.shape
                        if i2-i1 != h+offset or j2-j1 != w+offset:
                            img_pat = F.interpolate(img_pat, size= (h+offset, w+offset), mode='bilinear', align_corners=False)
                            mask_pat = cv2.resize(mask_pat, (w+offset, h+offset), cv2.INTER_AREA)
                            seg_pat = F.interpolate(seg_pat, size=(h+offset, w+offset), mode='bilinear', align_corners=False)
                            images = model(img_pat, seg_pat)
                            prob_map_full = images['pred_224'].data.cpu().numpy()[0,0]
                            pred_list.append(prob_map_full)
                            gt_list.append(mask_pat)
                            prob_map_full = cv2.resize(prob_map_full, (ori_shape[1], ori_shape[0]), cv2.INTER_AREA)
                        else:
                            images = model(img_pat, seg_pat)
                            prob_map_full = images['pred_224'].data.cpu().numpy()[0,0]
                            pred_list.append(prob_map_full)
                            gt_list.append(mask_pat)
                        # print(seg_pat.shape)
                        # print(prob_map_full.shape)
                        img_1[i1:i2 + offset, j1:j2 + offset] += prob_map_full
    img_1[img_1 > 1] = 1
    pred_mask = (img_1 * 255).astype(np.uint8)
    cv2.imwrite(filename=os.path.join(para['output'], f'{path.stem}.png'), img=pred_mask)

    if para['type'] == 'metric':
        for i in range(1, 10):
                threshold = i / 10
                metric = calc_metric(pred_list, gt_list, mode='list', threshold=threshold)
                print(metric)
                metric['accuracy'] = metric['accuracy'] / len(paths)
                metric['precision'] = metric['precision'] / len(paths)
                metric['recall'] = metric['recall'] / len(paths)
                metric['f1'] = metric['f1'] / len(paths)
                if len(metrics) < i:
                    metrics.append(metric)
                else:
                    metrics[i-1]['accuracy'] += metric['accuracy']
                    metrics[i-1]['precision'] += metric['precision']
                    metrics[i-1]['recall'] += metric['recall']
                    metrics[i-1]['f1'] += metric['f1']

if para['type'] == 'metric':
    print(metrics)
    d = datetime.today()
    datetime.strftime(d,'%Y-%m-%d %H-%M-%S')
    os.makedirs('./result_dir', exist_ok=True)
    with open(os.path.join('./result_dir', str(d)+'.txt'), 'a', encoding='utf-8') as fout:
            fout.write(para['model']+'\n')
            for i in range(1, 10): 
                    line =  "threshold:{:d} | accuracy:{:.5f} | precision:{:.5f} | recall:{:.5f} | f1:{:.5f} " \
                        .format(i, metrics[i-1]['accuracy'],  metrics[i-1]['precision'],  metrics[i-1]['recall'],  metrics[i-1]['f1']) + '\n'
                    fout.write(line)


print('Time taken: %.1f s' % (time.time() - epoch_start_time))