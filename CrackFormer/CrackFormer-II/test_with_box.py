from utils.utils import *
from utils.newValidator import *
from utils.Crackloader import *
from nets.crackformerII import crackformer
from nets.SDDNet import SDDNet
from nets.STRNet import STRNet

import sys
sys.path.append('/home/wj/local/crack_segmentation')
from metric import *
import os
import numpy as np
from pathlib import Path
import cv2 as cv
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import argparse
from os.path import join
from PIL import Image
import gc
from tqdm import tqdm

input_size = (448, 448)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # /nfs/DamDetection/data/  /mnt/hangzhou_116_homes/DamDetection/data/  /home/wj/dataset/crack/
    parser.add_argument('--img_dir',type=str, default='/mnt/nfs/wj/data', help='input dataset directory')
    parser.add_argument('--model_path', type=str, default='model/crackformer_epoch(68).pth', help='trained model path')
    parser.add_argument('--model_type', type=str, default='crackformer', choices=['crackformer', 'SDDNet', 'STRNet'])
    parser.add_argument('--out_pred_dir', type=str, default='./test_result', required=False,  help='prediction output dir')
    parser.add_argument('--type', type=str, default='metric' , choices=['out', 'metric'])
    args = parser.parse_args()

    if args.out_pred_dir != '':
        os.makedirs(args.out_pred_dir, exist_ok=True)
        for path in Path(args.out_pred_dir).glob('*.*'):
            os.remove(str(path))

    if args.model_type == 'crackformer':
        model = crackformer()
    elif args.model_type  == 'SDDNet':
        model = SDDNet(3, 1)
    elif args.model_type  == 'STRNet':
        model = STRNet(3, 1)
    else:
        print('undefind model name pattern')
        exit()
    model.load_state_dict(torch.load(args.model_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.type == 'out':
        DIR_IMG = os.path.join(args.img_dir, 'image')
        DIR_GT = ''
    elif args.type  == 'metric':
        DIR_IMG = os.path.join(args.img_dir, 'image')
        DIR_GT = os.path.join(args.img_dir, 'new_label')
    else:
        print('undefind test pattern')
        exit()

    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]

    paths = [path for path in Path(DIR_IMG).glob('*.*')]
    metrics=[]
    valid_log_dir = "./log/" + args.model_type + '/'
    best_model_dir = "./model/" + args.model_type + "/"
    validator = Validator(model, valid_log_dir, best_model_dir)
    for path in tqdm(paths):
        print(path)

        pred_list = []
        gt_list = []
        test_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(channel_means, channel_stds)])
        img_0 = cv.imread(str(path), 1)
        img_0 = np.asarray(img_0)
        if len(img_0.shape) != 3:
            print(f'incorrect image shape: {path.name}{img_0.shape}')
            continue

        img_0 = img_0[:,:,:3]
        img_height, img_width, *img_channels = img_0.shape

        if DIR_GT != '':
            mask_path = os.path.join(DIR_GT, path.stem+'.png')
            lab = cv.imread(mask_path, 0)
        else:
            lab = np.zeros(img_height, img_width)

        img_1 = np.zeros((img_height, img_width))
        cof = 1
        w, h = int(cof * input_size[0]), int(cof * input_size[1])
        offset = 32

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
                    # print(i1, i2, j1, j2)
                    img_pat = img_0[i1:i2 + offset, j1:j2 + offset]
                    ori_shape = img_pat.shape
                    # print(ori_shape)
                    if img_pat.shape != (h+offset, w+offset):
                        img_pat = cv.resize(img_pat, (w+offset, h+offset), cv.INTER_AREA)
                        # print(img_pat.shape)
                        prob_map_full = validator.validate(img_pat)
                        prob_map_full = cv.resize(prob_map_full, (ori_shape[1], ori_shape[0]), cv.INTER_AREA)
                    else:
                        prob_map_full = validator.validate(img_pat)
                    # print(prob_map_full.shape)
                    img_1[i1:i2 + offset, j1:j2 + offset] += prob_map_full
        img_1[img_1 > 1] = 1
        if args.out_pred_dir != '':
        #     # img_1[img_1 > 0.3] = 1
        #     # img_1[img_1 <= 0.3] = 0
            cv.imwrite(filename=join(args.out_pred_dir, f'{path.stem}.jpg'), img=(img_1 * 255).astype(np.uint8))

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
            img_pat = img_1[y1:y2,x1:x2]
            gt_pat = lab[y1:y2, x1:x2]
            ori_shape = gt_pat.shape
            img_pat = cv.resize(img_pat, (128, 128), cv.INTER_AREA)
            gt_pat = cv.resize(gt_pat, (128, 128), cv.INTER_AREA)
            pred_list.append(img_pat)
            gt_list.append(gt_pat)



        if args.type == 'metric':
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
        gc.collect()
    print(metrics)
    if args.type == 'metric':
        d = datetime.today()
        datetime.strftime(d,'%Y-%m-%d %H-%M-%S')
        os.makedirs('./result_dir', exist_ok=True)
        with open(os.path.join('./result_dir', str(d)+'.txt'), 'a', encoding='utf-8') as fout:
                fout.write(args.model_path+'\n')
                for i in range(1, 10): 
                    line =  "threshold:{:d} | accuracy:{:.5f} | precision:{:.5f} | recall:{:.5f} | f1:{:.5f} " \
                        .format(i, metrics[i-1]['accuracy'],  metrics[i-1]['precision'],  metrics[i-1]['recall'],  metrics[i-1]['f1']) + '\n'
                    fout.write(line)

