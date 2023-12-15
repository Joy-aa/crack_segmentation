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
input_size=[448, 448]

class Parser():
    def parse(self):
        self.default = HyperParameters()
        self.default.parse(unknown_arg_ok=True)

        parser = ArgumentParser()

        parser.add_argument('--dir', default='/mnt/nfs/wj/data/image', help='Directory with testing images')
        parser.add_argument('--seg', default='/mnt/hangzhou_116_homes/wj/result/0825/unet1/result_img448', help='Directory with testing images')
        parser.add_argument('--model',default='/home/wj/local/crack_segmentation/CascadePSP/weights/1_2023-08-24_07:23:11/model_300690', help='Pretrained model')
        parser.add_argument('--output', default='/home/wj/local/crack_segmentation/CascadePSP/results', help='Output directory')

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
        DIR_PRED = os.path.join(para['dir'], 'test_result')
        DIR_GT = ''
elif para['type']  == 'metric':
        DIR_IMG = os.path.join(para['dir'], 'image')
        DIR_PRED = os.path.join(para['dir'], 'test_result')
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
    else:
            lab = np.zeros((img_height, img_width))

    if DIR_PRED != '':
            seg_path = os.path.join(DIR_PRED, path.stem+'.jpg')
            seg = cv2.imread(seg_path, 0)
    else:
            seg = np.zeros((img_height, img_width))
    # img_0 = im_transform(img_0).unsqueeze(0).cuda()
    # print(img_0.shape)
    # seg = seg_transform(seg).unsqueeze(0).cuda()
    # print(seg.shape)
    filepath = os.path.join('/mnt/hangzhou_116_homes/DamDetection/data/result-stride_0.7/Jun02_06_33_42/box', path.stem+'.txt')
    boxes = []
    with open(filepath, 'r', encoding='utf-8') as f:
            for data in f.readlines():
                box = data.split(' ')[:-1]
                boxes.append(box)
    for box in boxes:
            x1, y1, x2, y2 = box
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            img_pat = img_0[y1:y2,x1:x2]
            mask_pat = lab[y1:y2,x1:x2]
            seg_pat = seg[y1:y2,x1:x2]
            ori_shape = seg_pat.shape
            if y2-y1 != 64 or x2 -x1 != 64:
                            img_pat = cv2.resize(img_pat, (64, 64), cv2.INTER_AREA)
                            mask_pat = cv2.resize(mask_pat, (64, 64), cv2.INTER_AREA)
                            seg_pat = cv2.resize(seg_pat, (64, 64), cv2.INTER_AREA)
            with torch.no_grad():
                input_img = im_transform(img_pat).unsqueeze(0).cuda()
                input_seg = seg_transform(seg_pat).unsqueeze(0).cuda()
                images = model(input_img, input_seg)
                prob_map_full = images['pred_224'].data.cpu().numpy()[0,0]
                pred_list.append(prob_map_full)
                gt_list.append(mask_pat)
            if prob_map_full.shape != ori_shape:
                prob_map_full = cv2.resize(prob_map_full, (ori_shape[1], ori_shape[0]), cv2.INTER_AREA)
            img_1[y1:y2,x1:x2] = prob_map_full
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