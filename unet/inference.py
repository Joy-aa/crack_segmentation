import sys
sys.path.append('/home/wj/local/crack_segmentation')
import os
import numpy as np
from pathlib import Path
import cv2 as cv
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from unet_transfer import UNet16, input_size
import matplotlib.pyplot as plt
import argparse
from os.path import join
from PIL import Image
import gc
from build_unet import load_unet_vgg16, load_unet_resnet_101, load_unet_resnet_34
from tqdm import tqdm
from metric import calc_metric

def evaluate_img(model, img):
    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(channel_means, channel_stds)])
    input_width, input_height = input_size[0], input_size[1]
    
    h, w = img.shape[0], img.shape[1]

    img_1 = cv.resize(img, (input_width, input_height), cv.INTER_AREA)
    X = train_tfms(Image.fromarray(img_1))
    X = Variable(X.unsqueeze(0)).cuda()  # [N, 1, H, W]

    mask = model(X)

    mask = F.sigmoid(mask[0, 0]).data.cpu().numpy()
    mask = cv.resize(mask, (w, h), cv.INTER_AREA)
    return mask

def evaluate_img_patch(model, img):
    input_width, input_height = input_size[0], input_size[1]

    img_height, img_width, img_channels = img.shape

    if img_width < input_width or img_height < input_height:
        return evaluate_img(model, img)

    stride_ratio = 0.1
    stride = int(input_width * stride_ratio)

    normalization_map = np.zeros((img_height, img_width), dtype=np.int16)

    patches = []
    patch_locs = []
    for y in range(0, img_height - input_height + 1, stride):
        for x in range(0, img_width - input_width + 1, stride):
            segment = img[y:y + input_height, x:x + input_width]
            normalization_map[y:y + input_height, x:x + input_width] += 1
            patches.append(segment)
            patch_locs.append((x, y))

    patches = np.array(patches)
    if len(patch_locs) <= 0:
        return None

    preds = []
    for i, patch in enumerate(patches):
        patch_n = train_tfms(Image.fromarray(patch))
        X = Variable(patch_n.unsqueeze(0)).cuda()  # [N, 1, H, W]
        masks_pred = model(X)
        mask = F.sigmoid(masks_pred[0, 0]).data.cpu().numpy()
        preds.append(mask)

    probability_map = np.zeros((img_height, img_width), dtype=float)
    for i, response in enumerate(preds):
        coords = patch_locs[i]
        probability_map[coords[1]:coords[1] + input_height, coords[0]:coords[0] + input_width] += response

    return probability_map

def disable_axis():
    plt.axis('off')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_ticklabels([])
    plt.gca().axes.get_yaxis().set_ticklabels([])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir',type=str, default='/mnt/hangzhou_116_homes/DamDetection/data', help='input dataset directory')
    parser.add_argument('--model_path', type=str, default='/home/wj/local/crack_segmentation/unet/checkpoints/model_epoch_19_vgg16.pt', help='trained model path')
    parser.add_argument('--model_type', type=str, default='vgg16', choices=['vgg16', 'resnet101', 'resnet34'])
    parser.add_argument('--out_pred_dir', type=str, default='./test_result', required=False,  help='prediction output dir')
    parser.add_argument('--threshold', type=float, default=0.1 , help='threshold to cut off crack response')
    args = parser.parse_args()

    if args.out_pred_dir != '':
        os.makedirs(args.out_pred_dir, exist_ok=True)
        for path in Path(args.out_pred_dir).glob('*.*'):
            os.remove(str(path))

    if args.model_type == 'vgg16':
        model = load_unet_vgg16(args.model_path)
    elif args.model_type  == 'resnet101':
        model = load_unet_resnet_101(args.model_path)
    elif args.model_type  == 'resnet34':
        model = load_unet_resnet_34(args.model_path)
        print(model)
    else:
        print('undefind model name pattern')
        exit()

    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    offset = 32

    DIR_IMG  = os.path.join(args.img_dir, 'image')
    DIR_MASK = os.path.join(args.img_dir, 'new_label')

    paths  = [path for path in Path(DIR_IMG).glob('*.*')]
    metrics = {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
    }
    for path in tqdm(paths):
        pred_list=[]
        gt_list = []
        #print(str(path))

        train_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(channel_means, channel_stds)])

        # img_0 = Image.open(str(path))
        img_0 = cv.imread(str(path), 1)
        img_0 = np.asarray(img_0)
        if len(img_0.shape) != 3:
            print(f'incorrect image shape: {path.name}{img_0.shape}')
            continue
        
        mask_path = os.path.join(DIR_MASK, path.stem+'.png')
        lab = cv.imread(mask_path, 0)

        img_0 = img_0[:,:,:3]

        img_height, img_width, img_channels = img_0.shape

        img_1 = np.zeros((img_height, img_width))

        cof = 1
        w, h = int(cof * input_size[0]), int(cof * input_size[1])
        # w = w - offset
        # h = h - offset

        a = 0
        if a == 1:
            img_1 = evaluate_img(model, img_0)
        else:
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
                    img_pat = img_0[i1:i2, j1:j2]
                    mask_pat = lab[i1:i2, j1:j2]
                    prob_map_full = evaluate_img(model, img_pat)
                    img_1[i1:i2, j1:j2] += prob_map_full

                    if i2-i1 != h or j2-j1 != w:
                        prob_map_full = cv.resize(prob_map_full, (w, h), cv.INTER_AREA)
                        mask_pat = cv.resize(mask_pat, (w, h), cv.INTER_AREA)

                    pred_list.append(prob_map_full)
                    gt_list.append(mask_pat)

        metric = calc_metric(pred_list, gt_list, mode='list', threshold=0.5)
        metrics['accuracy'] += metric['accuracy'] / len(paths)
        metrics['precision'] += metric['precision'] / len(paths)
        metrics['recall'] += metric['recall'] / len(paths)
        metrics['f1'] += metric['f1'] / len(paths)
        print(metric)
        img_1[img_1 > 1] = 1
        if args.out_pred_dir != '':
            cv.imwrite(filename=join(args.out_pred_dir, f'{path.stem}.jpg'), img=(img_1 * 255).astype(np.uint8))

        gc.collect()
    
    # metric = calc_metric(pred_list, gt_list, mode='list', threshold=0.5)
    with open('result.txt', 'a', encoding='utf-8') as fout:
            print(metrics)
            line =  "accuracy:{:.5f} | precision:{:.5f} | recall:{:.5f} | f1:{:.5f} " \
                .format(metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']) + '\n'
            fout.write(line)