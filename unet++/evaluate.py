import sys
import os
import numpy as np
from pathlib import Path
import cv2 as cv
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse
from os.path import join
from PIL import Image
import gc
from simple_Unetpp import UnetPlusPlus, load_model
from tqdm import tqdm

input_size = (448, 448)

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

def disable_axis():
    plt.axis('off')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_ticklabels([])
    plt.gca().axes.get_yaxis().set_ticklabels([])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir',type=str, default='../images', help='input dataset directory')
    parser.add_argument('--model_path', type=str, default='./checkpoints/Unet++_10.pth', help='trained model path')
    parser.add_argument('--out_pred_dir', type=str, default='./result_images', required=False,  help='prediction output dir')
    parser.add_argument('--threshold', type=float, default=0.1 , help='threshold to cut off crack response')
    args = parser.parse_args()

    if args.out_pred_dir != '':
        os.makedirs(args.out_pred_dir, exist_ok=True)
        for path in Path(args.out_pred_dir).glob('*.*'):
            os.remove(str(path))

    
    model = load_model(model_path = args.model_path, num_classes = 1)

    offset = 0
    paths = [path for path in Path(args.img_dir).glob('*.*')]
    for path in tqdm(paths):

        # img_0 = Image.open(str(path))
        img_0 = cv.imread(str(path), 1)
        img_0 = np.asarray(img_0)
        if len(img_0.shape) != 3:
            print(f'incorrect image shape: {path.name}{img_0.shape}')
            continue

        img_0 = img_0[:,:,:3]

        img_height, img_width, img_channels = img_0.shape

        img_1 = np.zeros((img_height, img_width))

        cof = 1
        w, h = int(cof * input_size[0]), int(cof * input_size[1])

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
                    print(i1, i2, j1, j2)
                    img_pat = img_0[i1:i2 + offset, j1:j2 + offset]
                    prob_map_full = evaluate_img(model, img_pat)
                    img_1[i1:i2 + offset, j1:j2 + offset] += prob_map_full
        img_1[img_1 > 1] = 1
        if args.out_pred_dir != '':
            cv.imwrite(filename=join(args.out_pred_dir, f'{path.stem}.jpg'), img=(img_1 * 255).astype(np.uint8))

        
        gc.collect()