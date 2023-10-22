import sys
sys.path.append('/home/wj/local/crack_segmentation')
from metric import *
import os
import numpy as np
from pathlib import Path
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from unet_transfer import UNet16
import argparse
from os.path import join
from PIL import Image
import gc
from tqdm import tqdm
import matplotlib.image as mpimg


def evaluate_img(model, img):
    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    test_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(channel_means, channel_stds)])
    X = test_tfms(Image.fromarray(img))
    X = Variable(X.unsqueeze(0)).cuda()  # [N, 1, H, W]

    mask = model(X)

    mask = torch.sigmoid(mask[0, 0]).data.cpu().numpy()
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
    
    mask[mask>threshold] = 1
    mask[mask<= threshold] = 0
    # mask[mask > 1] = 1
    return mask * 255

if __name__ == '__main__':
    model_path = '/home/wj/local/crack_segmentation/unet/checkpoints/stage2/model_best.pt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    #dam
    image_dir = '/nfs/wj/data/image/'
    label_dir = '/nfs/wj/data/new_label/'
    box_dir = '/nfs/wj/result-stride_0.7/Jun02_06_33_42/box/'
    ref_dir = '/nfs/wj/result-stride_0.7/Jun02_06_33_42/image/'
    res_dir = './result/stage2'
    save_dir = './invest/predict_dir'
    
    #dam2
    # image_dir = '/nfs/wj/data/dataV2/image'
    # label_dir = ''
    # box_dir = '/nfs/DamDetection/data/dataV2/result/Jun02_06_33_42/box/'
    # res_dir = './result/stage2V2/'
    # log_save = './result_dir'

    # model = UNet16(pretrained=True)
    
    # checkpoint = torch.load(model_path)
    # weights = checkpoint['model']
    # weights_dict = {}
    # for k, v in weights.items():
    #     new_k = k.replace('module.', '') if 'module' in k else k
    #     weights_dict[new_k] = v
    # model.load_state_dict(weights_dict)
    # model.cuda()
    
    if save_dir != '':
        os.makedirs(save_dir, exist_ok=True)

    paths = [path for path in Path(image_dir).glob('*.*')]
    metrics=[]
    for path in tqdm(paths):
        print(path)
        img_0 = cv.imread(str(path), 1)
        img_0 = np.asarray(img_0)
        txt_path = os.path.join(box_dir,path.stem+'.txt')

        # mask = single_result_no_box(model,img_0)
        # mask =  single_result(model=model,img=img_0,txt_path=txt_path)
        
        mask = cv.imread(os.path.join(res_dir,path.stem+'.png'), 0)
        gt = cv.imread(os.path.join(label_dir,path.stem+'.png'), 0)
        
        with open(os.path.join(save_dir, path.stem+'.txt'), 'a', encoding='utf-8') as fout:
            fout.write(str(path)+'\n')
            for i in range(1, 10):
                threshold = i / 10
                metric = calc_metric(mask, gt, mode='list', threshold=threshold, max_value=255)
                line =  "threshold:{:d} | accuracy:{:.5f} | precision:{:.5f} | recall:{:.5f} | f1:{:.5f} " \
                    .format(i, metric['accuracy'],  metric['precision'],  metric['recall'],  metric['f1']) + '\n'
                fout.write(line)
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
        
        # 4-channels result for label
        # mask[mask>127] = 255
        # gt[gt>0] = 255
        # image = cv.imread(os.path.join(ref_dir,path.stem+'.jpg'), 1)
        # mask = np.expand_dims(mask,axis=2)
        # zeros = np.zeros(mask.shape)
        # mask = np.concatenate((mask,zeros,zeros),axis=-1).astype(np.uint8)
        
        # label = np.expand_dims(gt,axis=2)
        # label = np.concatenate((zeros,zeros,label),axis=-1).astype(np.uint8)
        
        # temp = cv.addWeighted(label,1,mask,1,0)
        # res = cv.addWeighted(image,0.6,temp,0.4,0)

        # cv.imwrite(os.path.join(save_dir,path.stem+'.png'), res)
    gc.collect()
    with open(os.path.join(save_dir, 'total.txt'), 'a', encoding='utf-8') as fout:
        for i in range(1, 10): 
            line =  "threshold:{:d} | accuracy:{:.5f} | precision:{:.5f} | recall:{:.5f} | f1:{:.5f} " \
                .format(i, metrics[i-1]['accuracy'],  metrics[i-1]['precision'],  metrics[i-1]['recall'],  metrics[i-1]['f1']) + '\n'
            fout.write(line)

