import sys
sys.path.append('/home/wj/local/crack_segmentation')
from segtool.metric import *
import os
import numpy as np
from pathlib import Path
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import argparse
from os.path import join
from PIL import Image
import gc
from nets.DSCNet import DSCNet
from modeling.deeplab import DeepLab
from tqdm import tqdm
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0])) 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count())


def evaluate_img(model, img):
    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    test_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(channel_means, channel_stds)])
    X = test_tfms(Image.fromarray(img))
    X = Variable(X.unsqueeze(0)).cuda()  # [N, 1, H, W]

    pred = model(X)
    n, c, h, w = pred.size()
    temp_inputs = torch.softmax(pred.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    preds = torch.gt(temp_inputs[...,1], temp_inputs[...,0]).int().squeeze(-1)
    mask = preds.squeeze(0).view(h,w).contiguous().cpu().numpy()

    # mask = torch.sigmoid(mask[0, 0]).data.cpu().numpy()
    return mask

def single_result(model,img,txt_path,patch_size=64,threshold=0.5):
    h, w, *_ = img.shape
    mask = np.zeros((h, w))
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            data = line.split(' ')
            
            xmin, ymin, xmax, ymax = list(map(int, data[:4]))
            offset = 2 * patch_size + 64
            dx = xmin - patch_size
            if dx < 0:
                dx = 0
            if xmin + 64 + patch_size >= w:
                dx = w - offset

            dy = ymin - patch_size
            if dy < 0:
                dy = 0
            if ymin + 64 + patch_size >= h:
                dy = h - offset

            cut_img = img[dy:dy+offset, dx:dx+offset]
            cut_mask = evaluate_img(model,cut_img)
            
            mask[dy:dy+offset, dx:dx+offset] += cut_mask

    mask[mask > 1] = 1
    return mask * 255

if __name__ == '__main__':
    # model_path = '/home/wj/local/crack_segmentation/deeplab_drn/deeplabv3+_best_value.pth'
    model_path = '/home/wj/local/crack_segmentation/deeplab_drn/DSCNet_best_value.pth'
    #dam
    # image_dir = '/mnt/nfs/wj/data/image/'
    # label_dir = '/mnt/nfs/wj/data/new_label/'
    # ref_dir = '/mnt/nfs/wj/result-stride_0.7/Jun02_06_33_42/image/'
    # box_dir = '/mnt/nfs/wj/result-stride_0.7/Jun02_06_33_42/box/'
    # res_dir = './result/datav1'
    
    #dam2
    image_dir = '/mnt/nfs/wj/dataV2/image/'
    label_dir = '/mnt/nfs/wj/dataV2/label'
    box_dir = '/mnt/nfs/wj/dataV2/result/Jun02_06_33_42/box/'
    ref_dir = '/mnt/nfs/wj/dataV2/result/Jun02_06_33_42/image/'
    res_dir = './result/dscnet_dataV2/'

    # model = DeepLab(backbone='drn', output_stride=16)
    model =  DSCNet(3, 2, 15, 1.0, True, device, 16, 1)
    model.eval()
    model = model.cuda()
    # model = torch.nn.DataParallel(model)
    
    model.load_state_dict(torch.load(model_path,  map_location=device))

    # weights = torch.load(model_path,  map_location=device)
    # weights_dict = {}
    # for k, v in weights.items():
    #     new_k = k.replace('module.', '') if 'module' in k else k
    #     weights_dict[new_k] = v
    # model.load_state_dict(weights_dict)

    save_dir = os.path.join(res_dir, 'box')
    img_dir = os.path.join(res_dir, 'imgs')
    
    if save_dir != '':
        os.makedirs(save_dir, exist_ok=True)

    if img_dir != '':
        os.makedirs(img_dir, exist_ok=True)


    # channel_means = [0.485, 0.456, 0.406]
    # channel_stds  = [0.229, 0.224, 0.225]

    paths = [path for path in Path(image_dir).glob('*.*')]
    metrics = {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
    }
    ignore_list=[]
    for path in tqdm(paths):
        print(path)
        if str(path.stem) in ignore_list:
           continue
        img_0 = cv.imread(str(path), 1)
        img_0 = np.asarray(img_0)
        gt = cv.imread(os.path.join(label_dir,path.stem+'.png'), 0)

        txt_path = os.path.join(box_dir,path.stem+'.txt')
        mask =  single_result(model=model,img=img_0,txt_path=txt_path)

        # mask = cv.imread(os.path.join(res_dir,path.stem+'.png'), 0)

        gt = cv.imread(os.path.join(label_dir,path.stem+'.png'), 0)

        pred_list=[]
        gt_list=[]
        with open(txt_path, 'r') as f:
            h, w = gt.shape
            patch_size=64
            for line in f.readlines():
                data = line.split(' ')
                
                xmin, ymin, xmax, ymax = list(map(int, data[:4]))
                offset = 2 * patch_size + 64
                dx = xmin - patch_size
                if dx < 0:
                    dx = 0
                if xmin + 64 + patch_size >= w:
                    dx = w - offset

                dy = ymin - patch_size
                if dy < 0:
                    dy = 0
                if ymin + 64 + patch_size >= h:
                    dy = h - offset

                cut_gt= gt[dy:dy+offset, dx:dx+offset]
                cut_mask = mask[dy:dy+offset, dx:dx+offset]
                gt_list.append(cut_gt)
                pred_list.append(cut_mask)
        
        with open(os.path.join(save_dir, path.stem+'.txt'), 'a', encoding='utf-8') as fout:
            fout.write(str(path)+'\n')
            # for i in range(1, 10):
            #     threshold = i / 10
            if(len(pred_list) > 0):
                metric = calc_metric(pred_list, gt_list, mode='list', threshold=0.5, max_value=255)
                line =  "accuracy:{:.5f} | precision:{:.5f} | recall:{:.5f} | f1:{:.5f} " \
                .format(metric['accuracy'],  metric['precision'],  metric['recall'],  metric['f1']) + '\n'
            else:
                line =  "detection nothing! " + '\n'
            fout.write(line)
            metrics['accuracy'] += metric['accuracy'] / (len(paths) - len(ignore_list))
            metrics['precision'] += metric['precision'] / (len(paths) - len(ignore_list))
            metrics['recall'] += metric['recall'] / (len(paths) - len(ignore_list))
            metrics['f1'] += metric['f1'] / (len(paths) - len(ignore_list))
        
        # 4-channels result for label
        mask[mask>int(255*0.7)] = 255
        gt[gt>0] = 255
        image = cv.imread(os.path.join(ref_dir,path.stem+'.jpg'), 1)
        mask = np.expand_dims(mask,axis=2)
        zeros = np.zeros(mask.shape)
        mask = np.concatenate((mask,zeros,zeros),axis=-1).astype(np.uint8)
        
        label = np.expand_dims(gt,axis=2)
        label = np.concatenate((zeros,zeros,label),axis=-1).astype(np.uint8)
        
        temp = cv.addWeighted(label,1,mask,1,0)
        res = cv.addWeighted(image,0.6,temp,0.4,0)

        cv.imwrite(os.path.join(img_dir,path.stem+'.png'), res)
    gc.collect()
    with open(os.path.join(save_dir, 'total.txt'), 'a', encoding='utf-8') as fout:
        # for i in range(1, 10): 
            line =  "accuracy:{:.5f} | precision:{:.5f} | recall:{:.5f} | f1:{:.5f} " \
                .format(metrics['accuracy'],  metrics['precision'],  metrics['recall'],  metrics['f1']) + '\n'
            fout.write(line)
