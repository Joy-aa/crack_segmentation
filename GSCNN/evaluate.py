import sys
sys.path.append("/home/wj/local/crack_segmentation")
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
from tqdm import tqdm
import datetime
import csv
from segtool.metric import calc_metric
import network
import loss
from datasets import crack

def evaluate_img(model, img, test_tfms):
    X = test_tfms(Image.fromarray(img))
    X = Variable(X.unsqueeze(0)).cuda()  # [N, 1, H, W]

    pred, edge = model(X)

    n, c, h, w = pred.size()
    temp_inputs = torch.softmax(pred.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    preds = torch.gt(temp_inputs[...,1], temp_inputs[...,0]).int().squeeze(-1)
    mask = preds.squeeze(0).view(h,w).contiguous().cpu().numpy().astype('uint8')

    return mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # /nfs/DamDetection/data/  /mnt/hangzhou_116_homes/DamDetection/data/  /home/wj/dataset/crack/
    parser.add_argument('--joint_edgeseg_loss', action='store_true', default=True,help='joint loss')
    parser.add_argument('--img_wt_loss', action='store_true', default=False,help='per-image class-weighted loss')
    parser.add_argument('--edge_weight', type=float, default=1.0,help='Edge loss weight for joint loss')
    parser.add_argument('--seg_weight', type=float, default=1.0,help='Segmentation loss weight for joint loss')
    parser.add_argument('--att_weight', type=float, default=0,help='Attention loss weight for joint loss')
    parser.add_argument('--dual_weight', type=float, default=0,help='Dual loss weight for joint loss')
    parser.add_argument('--arch', type=str, default='network.gscnn.GSCNN')
    parser.add_argument('--trunk', type=str, default='resnet50', help='trunk model, can be: resnet101 (default), resnet50')
    parser.add_argument('-wb', '--wt_bound', type=float, default=1.0)
    parser.add_argument('--sgd', action='store_true', default=True)
    parser.add_argument('--sgd_finetuned',action='store_true',default=False)
    parser.add_argument('--adam', action='store_true', default=False)
    parser.add_argument('--amsgrad', action='store_true', default=False)
    parser.add_argument('--lr_schedule', type=str, default='poly',help='name of lr schedule: poly')
    parser.add_argument('--poly_exp', type=float, default=1.0,help='polynomial LR exponent')
    parser.add_argument('--snapshot', type=str, default='/home/wj/local/crack_segmentation/GSCNN/checkpoints/pretrain/gscnn_initial_epoch_50.pt')
    parser.add_argument('--restore_optimizer', action='store_true', default=False)

    parser.add_argument('--img_dir',type=str, default='/mnt/hangzhou_116_homes/DamDetection/data', help='input dataset directory')
    parser.add_argument('--model_path', type=str, default="/mnt/nfs/wj/checkpoints/gscnn/fineSeg/model_best.pt", help='trained model path')
    parser.add_argument('--type', type=str, default='metric' , choices=['out', 'metric'])
    parser.add_argument('--eval_type', type=str, default='512x512' , choices=['512x512', 'test_with_box_192'])
    args = parser.parse_args()
    args.dataset_cls = crack
    out_pred_dir = os.path.join(os.path.dirname(os.path.abspath(args.model_path)), 'data')
    if out_pred_dir != '':
        os.makedirs(out_pred_dir, exist_ok=True)
        for path in Path(out_pred_dir).glob('*.*'):
            os.remove(str(path))

    criterion, criterion_val = loss.get_loss(args)
    model = network.get_net(args, criterion)
    state = torch.load(args.model_path)
    model.load_state_dict(state['state_dict'])
    # weights = state['model']
    # weights_dict = {}
    # for k, v in weights.items():
    #     new_k = k.replace('module.', '') if 'module' in k else k
    #     weights_dict[new_k] = v
    # model.load_state_dict(weights_dict)
    # model.cuda()
    model.eval()
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
    metrics = {
        'accuracy': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0,
    }
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
            lab = np.zeros((img_height, img_width))

        img_1 = np.zeros((img_height, img_width))

        cof = 1
        input_size = (480, 480)
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
                    mask_pat = lab[i1:i2 + offset, j1:j2 + offset]
                    ori_shape = mask_pat.shape
                    # print(ori_shape)
                    if mask_pat.shape != (h+offset, w+offset):
                        img_pat = cv.resize(img_pat, (w+offset, h+offset), cv.INTER_AREA)
                        mask_pat = cv.resize(mask_pat, (w+offset, h+offset), cv.INTER_AREA)
                        # print(img_pat.shape)
                        prob_map_full = evaluate_img(model, img_pat, test_tfms)
                        pred_list.append(prob_map_full)
                        gt_list.append(mask_pat)
                        prob_map_full = cv.resize(prob_map_full, (ori_shape[1], ori_shape[0]), cv.INTER_AREA)
                    else:
                        prob_map_full = evaluate_img(model, img_pat,test_tfms)
                        pred_list.append(prob_map_full)
                        gt_list.append(mask_pat)
                    img_1[i1:i2 + offset, j1:j2 + offset] += prob_map_full
        img_1[img_1 > 1] = 1
        if out_pred_dir != '':
            mask = np.expand_dims(img_1, axis=0)
            label = np.expand_dims(lab, axis=0)
            mask = (mask.transpose(1, 2, 0)*255).astype('uint8')
            label = label.transpose(1, 2, 0).astype('uint8')
            # image = img_0.transpose(1, 0, 2).astype(np.uint8)
            zeros = np.zeros(mask.shape)
            mask = np.concatenate((mask,zeros,zeros),axis=-1).astype(np.uint8)
            label = np.concatenate((zeros,zeros,label),axis=-1).astype(np.uint8)

            temp = cv.addWeighted(label,1,mask,1,0)
            res = cv.addWeighted(img_0,0.6,temp,0.4,0)

            cv.imwrite(filename=join(out_pred_dir, f'{path.stem}.jpg'), img=res)

        if args.type == 'metric':
            # for i in range(1, 10):
                # threshold = i / 10
                threshold = 0.5
                metric = calc_metric(pred_list, gt_list, mode='list', threshold=threshold)
                print(metric)
                metric['accuracy'] = metric['accuracy'] / len(paths)
                metric['precision'] = metric['precision'] / len(paths)
                metric['recall'] = metric['recall'] / len(paths)
                metric['f1'] = metric['f1'] / len(paths)
                # if len(metrics) < i:
                #     metrics.append(metric)
                # else:
                metrics['accuracy'] += metric['accuracy']
                metrics['precision'] += metric['precision']
                metrics['recall'] += metric['recall']
                metrics['f1'] += metric['f1']

        gc.collect()
    print(metrics)
    if args.type == 'metric':
        d = datetime.datetime.today()
        datetime.datetime.strftime(d,'%Y-%m-%d %H-%M-%S')
        os.makedirs('./result_dir', exist_ok=True)
        with open(os.path.join('./result_dir', str(d)+'.txt'), 'a', encoding='utf-8') as fout:
                fout.write(args.eval_type+'\n')
                fout.write(args.model_path+'\n')
                # for i in range(1, 10): 
                line =  "threshold:{:d} | accuracy:{:.5f} | precision:{:.5f} | recall:{:.5f} | f1:{:.5f} " \
                    .format(i, metrics['accuracy'],  metrics['precision'],  metrics['recall'],  metrics['f1']) + '\n'
                fout.write(line)