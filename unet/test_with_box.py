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
from unet.network.unet_transfer import UNet16
import argparse
from os.path import join
from PIL import Image
import gc
from unet.network.build_unet import load_unet_vgg16, load_unet_resnet_101, load_unet_resnet_34
from tqdm import tqdm


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
    
    # mask[mask>threshold] = 1
    # mask[mask<= threshold] = 0
    mask[mask > 1] = 1
    return mask * 255

if __name__ == '__main__':
    model_path = '/home/wj/local/crack_segmentation/unet/checkpoints/stage2/model_best.pt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    #dam
    # image_dir = '/nfs/wj/data/image/'
    # label_dir = '/nfs/wj/data/new_label/'
    # box_dir = '/nfs/wj/result-stride_0.7/Jun02_06_33_42/box/'
    # res_dir = './result/stage2'
    # log_save = './result_dir'
    
    #dam2
    image_dir = '/nfs/wj/data/dataV2/image'
    label_dir = ''
    box_dir = '/nfs/DamDetection/data/dataV2/result/Jun02_06_33_42/box/'
    res_dir = './result/stage2V2/'
    log_save = './result_dir'

    # model = load_unet_vgg16(args.model_path)screen -r unet

    model = UNet16(pretrained=True)
    # model.load_state_dict(torch.load(args.model_path))
    
    checkpoint = torch.load(model_path)
    weights = checkpoint['model']
    weights_dict = {}
    for k, v in weights.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v
    model.load_state_dict(weights_dict)
    model.cuda()
    
    if res_dir != '':
        os.makedirs(res_dir, exist_ok=True)


    # channel_means = [0.485, 0.456, 0.406]
    # channel_stds  = [0.229, 0.224, 0.225]

    paths = [path for path in Path(image_dir).glob('*.*')]
    metrics=[]
    for path in tqdm(paths):
        print(path)
        img_0 = cv.imread(str(path), 1)
        img_0 = np.asarray(img_0)
        # print(img_0)
        txt_path = os.path.join(box_dir,path.stem+'.txt')

        # mask = single_result_no_box(model,img_0)
        mask =  single_result(model=model,img=img_0,txt_path=txt_path)
        
        res = mask
        cv.imwrite(filename=join(res_dir, f'{path.stem}.png'), img=(res))

        if(label_dir != ''):
            gt = cv.imread(os.path.join(label_dir,path.stem+'.png'), 0)

            for i in range(1, 10):
                threshold = i / 10
                metric = calc_metric(mask, gt, mode='list', threshold=threshold, max_value=255)
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
    if(label_dir != ''):
        print(metrics)
        d = datetime.today()
        datetime.strftime(d,'%Y-%m-%d %H-%M-%S')
        os.makedirs('./result_dir', exist_ok=True)
        with open(os.path.join('./result_dir', str(d)+'.txt'), 'a', encoding='utf-8') as fout:
                fout.write(model_path+'\n')
                fout.write('test_with_box_192\n')
                for i in range(1, 10): 
                    line =  "threshold:{:d} | accuracy:{:.5f} | precision:{:.5f} | recall:{:.5f} | f1:{:.5f} " \
                        .format(i, metrics[i-1]['accuracy'],  metrics[i-1]['precision'],  metrics[i-1]['recall'],  metrics[i-1]['f1']) + '\n'
                    fout.write(line)

