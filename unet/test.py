import sys
sys.path.append("/home/wj/local/crack_segmentation")
from data_loader import ImgDataSet
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
from torch.utils.data import DataLoader, random_split
from unet_transfer import UNet16, UNet16V2
import argparse
from os.path import join
from PIL import Image
import gc
from build_unet import load_unet_vgg16, load_unet_resnet_101, load_unet_resnet_34
from tqdm import tqdm


def evaluate_img(model, img, test_tfms):
    X = test_tfms(Image.fromarray(img))
    X = Variable(X.unsqueeze(0)).cuda()  # [N, 1, H, W]

    mask = model(X)

    mask = F.sigmoid(mask[0, 0]).data.cpu().numpy()
    return mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # /nfs/DamDetection/data/  /mnt/hangzhou_116_homes/wj/data/  /home/wj/dataset/crack/
    parser.add_argument('--img_dir',type=str, default='../images', help='input dataset directory')
    parser.add_argument('--model_path', type=str, default='/mnt/hangzhou_116_homes/wj/model/unet/gelu_model_epoch_48.pt', help='trained model path')
    parser.add_argument('--model_type', type=str, default='vgg16', choices=['vgg16', 'vgg16V2', 'resnet101', 'resnet34'])
    args = parser.parse_args()


    if args.model_type == 'vgg16':
        # model = load_unet_vgg16(args.model_path)
        model = UNet16(pretrained=True)
        # model.load_state_dict(torch.load(args.model_path))
        
        checkpoint = torch.load(args.model_path)
        weights = checkpoint['model']
        weights_dict = {}
        for k, v in weights.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v
        model.load_state_dict(weights_dict)
        model.cuda()
    elif args.model_type == 'vgg16V2':
        model = UNet16V2(pretrained=True)
        checkpoint = torch.load(args.model_path)
        weights = checkpoint['model']
        weights_dict = {}
        for k, v in weights.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v
        model.load_state_dict(weights_dict)
        model.cuda()
    elif args.model_type  == 'resnet101':
        model = load_unet_resnet_101(args.model_path)
    elif args.model_type  == 'resnet34':
        model = load_unet_resnet_34(args.model_path)
        print(model)
    else:
        print('undefind model name pattern')
        exit()
    # model = nn.DataParallel(model)
    TRAIN_IMG  = os.path.join(args.img_dir, 'imgs')
    TRAIN_MASK = os.path.join(args.img_dir, 'masks')
    train_img_names  = [path.name for path in Path(TRAIN_IMG).glob('*.png')]
    train_mask_names = [path.name for path in Path(TRAIN_MASK).glob('*.png')]
    
    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose([transforms.ToTensor()])
                                    #  transforms.Normalize(channel_means, channel_stds)])
    val_tfms = transforms.Compose([transforms.ToTensor()])
                                #    transforms.Normalize(channel_means, channel_stds)])
    mask_tfms = transforms.Compose([transforms.ToTensor()])
    train_dataset = ImgDataSet(img_dir=TRAIN_IMG, img_fnames=train_img_names, img_transform=train_tfms, mask_dir=TRAIN_MASK, mask_fnames=train_mask_names, mask_transform=mask_tfms)
    _dataset, test_dataset = random_split(train_dataset, [0.9, 0.1],torch.Generator().manual_seed(42))
    test_loader = torch.utils.data.DataLoader(test_dataset, 1, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=4)
    print(f'total test images = {len(test_loader)}')

    metrics=[]
    pred_list = []
    gt_list = []
    bar = tqdm(total=len(test_loader))
    with torch.no_grad():
        for idx, (img, lab) in enumerate(test_loader, 1):
            val_data  = Variable(img).cuda()
            pred = model(val_data)
            pred_list.append(torch.sigmoid(pred.contiguous().cpu()).numpy())
            gt_list.append(lab.numpy())
            bar.update(1)
    bar.close

    for i in range(1, 10):
                threshold = i / 10
                metric = calc_metric(pred_list, gt_list, mode='list', threshold=threshold)
                print(metric)
                if len(metrics) < i:
                    metrics.append(metric)
                else:
                    metrics[i-1]['accuracy'] += metric['accuracy']
                    metrics[i-1]['precision'] += metric['precision']
                    metrics[i-1]['recall'] += metric['recall']
                    metrics[i-1]['f1'] += metric['f1']

    gc.collect()
    print(metrics)
    d = datetime.today()
    datetime.strftime(d,'%Y-%m-%d %H-%M-%S')
    os.makedirs('./result_dir', exist_ok=True)
    with open(os.path.join('./result_dir', str(d)+'.txt'), 'a', encoding='utf-8') as fout:
                fout.write(args.model_path+'\n')
                for i in range(1, 10): 
                    line =  "threshold:{:d} | accuracy:{:.5f} | precision:{:.5f} | recall:{:.5f} | f1:{:.5f} " \
                        .format(i, metrics[i-1]['accuracy'],  metrics[i-1]['precision'],  metrics[i-1]['recall'],  metrics[i-1]['f1']) + '\n'
                    fout.write(line)
