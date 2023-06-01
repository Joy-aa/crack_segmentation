from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import argparse
import pandas as pd
import csv
import datetime
import cv2 
import cv2 as cv
from PIL import Image, ImageEnhance
import os
from pathlib import Path
from torch.utils.data import random_split
from tqdm import tqdm

def tensorWriter(metric, value, iter):
    writer = SummaryWriter()
    for n_iter in range(100):
        writer.add_scalar('Loss/train', np.random.random(), n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

def getmask(prob_img, threshold = 0.5):
    mask = np.zeros(prob_img.shape, dtype='uint8')
    img=(prob_img * 255).astype(np.uint8)
    threshold = int(threshold * 255)
    mask[img > threshold] = 255
    return mask

def calc_accuracy(gt_mask, pred_mask):
    gt_mask_fla = gt_mask.flatten()
    print(gt_mask_fla.shape)
    pred_mask_fla = pred_mask.flatten()
    correct = np.sum(gt_mask_fla == pred_mask_fla)
    total = gt_mask_fla.shape[0]
    # print(gt_mask_fla.shape)
    return correct / total

def calc_precision(gt_mask, pred_mask):
    gt_mask_fla = gt_mask.flatten()
    pred_mask_fla = pred_mask.flatten()
    tind = np.where(pred_mask_fla == 255)
    tp = np.sum(pred_mask_fla[tind] == gt_mask_fla[tind])
    fp = np.sum(pred_mask_fla[tind] != gt_mask_fla[tind])
    total = tp+fp
    if total == 0:
        total = 0x3f3f3f3f
    return tp / total

def calc_recall(gt_mask, pred_mask):
    gt_mask_fla = gt_mask.flatten()
    pred_mask_fla = pred_mask.flatten()
    tind = np.where(pred_mask_fla == 255)
    find = np.where(pred_mask_fla == 0)
    tp = np.sum(pred_mask_fla[tind] == gt_mask_fla[tind])
    fn = np.sum(pred_mask_fla[find] != gt_mask_fla[find])
    total = tp+fn
    if total == 0:
        total = 0x3f3f3f3f
    return tp / total

def calc_f1(gt_mask, pred_mask):
    precision = calc_precision(gt_mask, pred_mask)
    recall = calc_recall(gt_mask, pred_mask)
    f1 = 2*precision*recall / (precision + recall)
    return f1
    
def miou(pred: torch.Tensor, mask: torch.Tensor):
    pred, mask = pred.flatten().long(), mask.flatten().long()
    correct = torch.bincount(mask[pred == mask], minlength=2)
    pred_count = torch.bincount(pred, minlength=2)
    total = torch.bincount(mask, minlength=2)
    union = pred_count + total - correct
    union[union == 0] = 1
    iou = correct / union
    return iou.mean()

def calc_miou(gt_mask, pred_mask):
    mask = gt_mask.flatten()
    pred = pred_mask.flatten()
    correct = np.bincount(mask[pred == mask], minlength=256)
    pred_count = np.bincount(pred, minlength=256)
    total = np.bincount(mask, minlength=256)
    union = pred_count + total - correct
    union[union == 0] = 1
    iou = correct / union
    return iou.mean()

def cv2_rotation(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

def cv2_trans(img, angle = 0, type = 0):
    if type == 0:
        return img
    if angle%180==0:
        return cv2.flip(img, 0)
    else:
        return cv2.flip(img, 1)

def cv2_trans_(img, type = 0):
    if type == 0:
        return cv2.transpose(img)
    if type == 1:
        return cv2.flip(img, 0)
    if type == 2:
        return cv2.flip(img, 1)
    if type == 3:
        return cv2.flip(img, -1)
    return img


# ---------------------PIL 实现-------------------
def rotation(img, angle):
    rotation_img = img.rotate(angle) # 旋转角度
    return rotation_img

def randomColor(image): # 图像增强
    # 亮度变化
    enh_bri = ImageEnhance.Brightness(image)
    brightness = np.random.randint(8, 13) / 10.  # 随机因子
    image1 = enh_bri.enhance(brightness)

    # 色度变化
    enh_col = ImageEnhance.Color(image1)
    color = np.random.randint(5, 25) / 10.  # 随机因子
    image2 = enh_col.enhance(color)

    # 对比度变化
    enh_con = ImageEnhance.Contrast(image2)
    contrast = np.random.randint(8, 25) / 10.  # 随机因子
    image3 = enh_con.enhance(contrast)

    # 锐度变化
    enh_sha = ImageEnhance.Sharpness(image3)
    sharpness = np.random.randint(5, 51) / 10.  # 随机因子
    image4 = enh_sha.enhance(sharpness)

    # random_factor = np.random.randint(9, 20) / 10.  # 随机因子
    # result = ImageEnhance.Contrast(img).enhance(random_factor)  # 调整图像对比度
    # random_factor = np.random.randint(0, 51) / 10.  # 随机因子
    # result = ImageEnhance.Sharpness(result).enhance(random_factor)  # 调整图像锐度

    return image4


# ---------------------获取文件夹下的所有图片-------------------
def list_dir(file_dir):
    '''
        通过 listdir 得到的是仅当前路径下的文件名，不包括子目录中的文件，如果需要得到所有文件需要递归
    '''
    dir_list = os.listdir(file_dir)
    result_list = []
    for cur_file in dir_list:
        # 获取文件的绝对路径
        path = os.path.join(file_dir, cur_file)
        if os.path.isfile(path): # 判断是否是文件还是目录需要用绝对路径
            result_list.append(path)
        if os.path.isdir(path):
            result_list += list_dir(path) # 递归子目录
    return result_list

def cut_data(root_img_path, root_label_path, save_img_path, save_label_path, size=(448,448)):

# root_img_path = 'data/img'
# root_label_path = 'data/label'

# save_img_path = 'cut_data/img'
# save_label_path = 'cut_data/label'

    list_img = os.listdir(root_img_path)

    w, h = size

    cut = 0

    for img_name in list_img:
        cut += 1
        print(cut)
        
        cnt = 1
        img_path = os.path.join(root_img_path, img_name)
        label_path = os.path.join(root_label_path, img_name)
        img = cv2.imread(img_path, 1)
        label = cv2.imread(label_path, 0)

        img_height, img_width = img.shape[0], img.shape[1]
        
        for i in range(0, img_height, h):
            for j in range(0, img_width, w):
                i1 = i
                i2 = i + h
                j1 = j
                j2 = j + w
                if i2>img_height:
                    i1 = max(img_height - h, 0)
                    i2 = img_height
                if j2>img_width:
                    j1 = max(img_width - w, 0)
                    j2 = img_width
                
                img_pat = img[i1:i2, j1:j2]
                label_pat = label[i1:i2, j1:j2]
                img_pat = cv2.resize(img_pat, (w, h))
                label_pat = cv2.resize(label_pat, (w, h))

                name, ext = img_name.split('.')
                save_name = name + f'_{cnt}.jpg'

                img_save_path = os.path.join(save_img_path, save_name)
                label_save_path = os.path.join(save_label_path, save_name)

                cv2.imwrite(img_save_path, img_pat)
                cv2.imwrite(label_save_path, label_pat)

                cnt += 1

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Get Evaluate Value')
    # parser.add_argument('--data_dir',type=str, help='input dataset directory')
    # /home/wj/dataset/seg_dataset
    # parser.add_argument('--model_info', type=str, default='vgg16')
 
    # header = ['datetime','modelType','trainLoss','validLoss','Accuracy','Precision','Recall',"F1-score",'MIOU']
    # today = str(datetime.date.today())
    # # row = [(now_time,'张三','98')]
    # with open(f'result_{today}.csv','a+',encoding='utf8',newline='') as f :
    #     writer = csv.writer(f)
    #     writer.writerow(header)
    #     # writer.writerows(row)

    # root_img_path = 'data/img'
    # root_label_path = 'data/label'

    # save_img_path = 'cut_data/img'
    # save_label_path = 'cut_data/label'

    data_dir = '/nfs/ymd/DamCrack'
    DIR_IMG  = os.path.join(data_dir, 'image')
    DIR_MASK = os.path.join(data_dir, 'label')

    img_names  = [path.name for path in Path(DIR_IMG).glob('*.jpg')]
    # mask_names = [path.name for path in Path(DIR_MASK).glob('*.bmp')]

    train_size = int(0.85*len(img_names))
    valid_size = len(img_names) - train_size
    train_names, valid_names = random_split(img_names, [train_size, valid_size])\
    
    with open('/home/wj/pycharmProjects/crack_segmentation/DeepCrack/codes/data/train_example.txt','w') as f:    #设置文件对象
        for name in train_names:
            str = os.path.join(DIR_IMG, name) + ' ' + os.path.join(DIR_MASK, name.split('.')[0] + '.bmp')
            # print(str)
            f.write(str + '\n')  


    with open('/home/wj/pycharmProjects/crack_segmentation/DeepCrack/codes/data/val_example.txt','w') as f:    #设置文件对象
        for name in valid_names:
            str = os.path.join(DIR_IMG, name) + ' ' + os.path.join(DIR_MASK, name.split('.')[0] + '.bmp')
            f.write(str + '\n')  
    # img_paths  = [path for path in Path(DIR_MASK).glob('noncrack*.bmp')]
    # # mask_paths = [path for path in Path(DIR_MASK).glob('Crack500*.jpg')]
    # # print(len(img_paths))
    # for img_path in tqdm(img_paths):
    #     img = cv.imread(str(img_path), 0)
    #     # print(img.shape)

    #     cv.imwrite(os.path.join(str(img_path).split('.')[0] + '.bmp'), img)
    #     os.remove(os.path.join(DIR_MASK, img_path.name))
    #     # print(os.path.join(str(img_path).split('.')[0] + '.bmp'))
    #     # print(os.path.join(DIR_MASK, img_path.name))



