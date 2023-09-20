from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from torch.autograd import Variable
from datasets import VOCSegmentation, Cityscapes, cityscapes
from torchvision import transforms as T
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
from pathlib import Path
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
from datetime import datetime

import sys
sys.path.append('/home/wj/local/crack_segmentation')
from metric import calc_metric

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--input", type=str, required=True,
                        help="path to a single image or image directory")
    parser.add_argument("--dataset", type=str, default='crack',
                        choices=['voc', 'cityscapes'], help='Name of training set')

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results_to", default=None,
                        help="save segmentation results to the specified dir")

    # parser.add_argument("--crop_val", action='store_true', default=False,
    #                     help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    # parser.add_argument("--crop_size", type=int, default=513)

    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser

def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
    elif opts.dataset.lower() == 'crack':
        opts.num_classes = 1

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    # image_files = []
    # if os.path.isdir(opts.input):
    #     for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
    #         files = glob(os.path.join(opts.input, '**/*.%s'%(ext)), recursive=True)
    #         if len(files)>0:
    #             image_files.extend(files)
    # elif os.path.isfile(opts.input):
    #     image_files.append(opts.input)
    
    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    #denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    DIR_IMG = os.path.join(opts.input, 'image')
    DIR_GT = os.path.join(opts.input, 'new_label')
    paths = [path for path in Path(DIR_IMG).glob('*.*')]
    metrics=[]
    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)
    with torch.no_grad():
        model = model.eval()
        for path in tqdm(paths):
            print(path)
            pred_list = []
            gt_list = []
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
                lab = np.zeros((img_height, img_width))

            img_1 = np.zeros((img_height, img_width))

            cof = 1
            input_size = (112, 112)
            w, h = int(cof * input_size[0]), int(cof * input_size[1])
            offset = 16

            torch.set_num_threads(1)
            torch.backends.cudnn.benchmark = True
            
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
                        img_pat = img_0[i1:i2 + offset, j1:j2 + offset]
                        mask_pat = lab[i1:i2 + offset, j1:j2 + offset]
                        ori_shape = mask_pat.shape
                        # print(ori_shape)
                        if mask_pat.shape != (h+offset, w+offset):
                            img_pat = cv.resize(img_pat, (w+offset, h+offset), cv.INTER_AREA)
                            mask_pat = cv.resize(mask_pat, (w+offset, h+offset), cv.INTER_AREA)
                            X = transform(Image.fromarray(img_pat))
                            X = Variable(X.unsqueeze(0)).cuda()  # [N, 1, H, W]
                            mask = model(X)
                            prob_map_full = torch.sigmoid(mask[0, 0]).data.cpu().numpy()
                            pred_list.append(prob_map_full)
                            gt_list.append(mask_pat)
                            prob_map_full = cv.resize(prob_map_full, (ori_shape[1], ori_shape[0]), cv.INTER_AREA)
                        else:
                            X = transform(Image.fromarray(img_pat))
                            X = Variable(X.unsqueeze(0)).cuda()  # [N, 1, H, W]
                            mask = model(X)
                            prob_map_full = torch.sigmoid(mask[0, 0]).data.cpu().numpy()
                            pred_list.append(prob_map_full)
                            gt_list.append(mask_pat)
                        img_1[i1:i2 + offset, j1:j2 + offset] += prob_map_full
            img_1[img_1 > 1] = 1
            if opts.save_val_results_to != '':
                cv.imwrite(filename=os.path.join(opts.save_val_results_to, f'{path.stem}.png'), img=(img_1 * 255).astype(np.uint8))


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

    print(metrics)
    d = datetime.today()
    datetime.strftime(d,'%Y-%m-%d %H-%M-%S')
    os.makedirs('./result_dir', exist_ok=True)
    with open(os.path.join('./result_dir', str(d)+'.txt'), 'a', encoding='utf-8') as fout:
                fout.write(opts.ckpt+'\n')
                for i in range(1, 10): 
                    line =  "threshold:{:d} | accuracy:{:.5f} | precision:{:.5f} | recall:{:.5f} | f1:{:.5f} " \
                        .format(i, metrics[i-1]['accuracy'],  metrics[i-1]['precision'],  metrics[i-1]['recall'],  metrics[i-1]['f1']) + '\n'
                    fout.write(line)

if __name__ == '__main__':
    main()
