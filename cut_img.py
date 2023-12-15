import os
import cv2
import codecs
from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as transforms
np.set_printoptions(threshold=np.inf)

def img_cut():
    return
    img_h, img_w, *_ = img.shape
    # print(img_h,img_w)

    # 224 * 224
    dx = int((1. - overlap)*slice_w)
    dy = int((1. - overlap)*slice_h)

    # 切割
    for y0 in range(0, img_h, dy):
        for x0 in range(0, img_w, dx):
            if y0 + slice_h > img_h:
                y = img_h - slice_h
            else:
                y = y0
            if x0 + slice_w > img_w:
                x = img_w - slice_w
            else:
                x = x0
            slice_xmax = x + slice_w
            slice_ymax = y + slice_h
            '''
                x,y: 左上角
                slice_xmax,slice_ymax: 右下角
            '''
            cut_img = img[y:slice_ymax, x:slice_xmax]
            cut_limg = limg[y:slice_ymax, x:slice_xmax]


def cut_overlap_img(root_path: str, raw_imgs_dir: str, raw_limgs_dir: str, out_imgs_dir: str, overlap=0.4, slice_size=224,threshold=1000):
    # if not os.path.exists(out_imgs_dir):
    #     os.makedirs(out_imgs_dir)
    # if not os.path.exists(os.path.join(out_imgs_dir, 'imgs')):
    #     os.makedirs(os.path.join(out_imgs_dir, 'imgs'))
    #     os.makedirs(os.path.join(out_imgs_dir, 'masks'))
        
    # raw_imgs_dir = os.path.join(root_path,raw_imgs_dir)
    # raw_limgs_dir = os.path.join(root_path,raw_limgs_dir)
    # 非裂缝图片数量
    cnt = 0
    # 裂缝图片数量
    

    for per_img in tqdm(os.listdir(raw_imgs_dir)):     
        limg_path = os.path.join(raw_limgs_dir, per_img.split('.')[0]+'.png')
        
        #crack500
        # limg_path = os.path.join(raw_limgs_dir, per_img.split('.')[0]+'_mask.png')
        
        img_path = os.path.join(raw_imgs_dir, per_img)
        if os.path.isdir(img_path) or per_img.split('.')[-1] == 'txt':
            continue
        # 图像读取
        img = cv2.imread(img_path)
        # limg 是标注图片
        limg = cv2.imread(limg_path, 0)
        if limg is None:
            continue 
        h, w, *_ = img.shape
        step =int((1-overlap) * slice_size)
        # 切割
        for x in range(0,w,step):
            for y in range(0,h,step):
                dx = x
                if x +  slice_size >= w:
                    dx = w - slice_size
                if dx < 0:
                    break
                
                dy = y
                if y + slice_size >= h:
                    dy = h - slice_size
                if dy < 0:
                    continue
                # print(dy, dy+offset, dx, dx+offset)
                cut_img = img[dy:dy+slice_size, dx:dx+slice_size]
                cut_limg = limg[dy:dy+slice_size, dx:dx+slice_size]
                # flag = int(judge(cut_img))
                # print((cut_img),(cut_limg))
                if np.count_nonzero(cut_limg) <= threshold:
                    continue
                outpath = os.path.join('image_{}.png'.format(cnt))
                outpath = per_img.split('.')[0] + '_{}_{}_{}_{}.png'.format(dy, dy+slice_size, dx, dx+slice_size)
                cnt += 1
                # print(img.shape,dy,dy+offset,dx,dx+offset)
                # print(np.count_nonzero(cut_limg))
                cut_limg[cut_limg!=0] = 255
                cv2.imwrite(os.path.join(
                    out_imgs_dir, 'imgs', outpath), cut_img)
                cv2.imwrite(os.path.join(
                    out_imgs_dir, 'masks', outpath), cut_limg)
        print(cnt)

def simple_slice(raw_txt_dir: str, raw_imgs_dir: str, raw_limgs_dir: str, raw_mimgs_dir: str, out_imgs_dir: str, patch_num=1):
    '''
     根据标注图片切割目录下的所有图片，并进行二分类
    '''
    # 非裂缝图片数量
    cnt = 0
    # 裂缝图片数量

    for per_img in tqdm(os.listdir(raw_imgs_dir)):
        # limg_name = per_img.split('.')[0]+'_mask.png'
        limg_path = os.path.join(raw_limgs_dir, per_img.split('.')[0]+'.png')
        # mimg_path = os.path.join(raw_mimgs_dir, per_img.split('.')[0]+'.jpg')
        img_path = os.path.join(raw_imgs_dir, per_img)
        txt_path = os.path.join(raw_txt_dir, per_img.split('.')[0]+'.txt')
        if os.path.isdir(img_path) or per_img.split('.')[-1] == 'txt':
            continue
        # 图像读取
        img = cv2.imread(img_path)
        # limg 是标注图片
        limg = cv2.imread(limg_path, 0)
        # mimg = cv2.imread(mimg_path, 0)
        h, w, *_ = img.shape
        # 切割
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                data = line.split(' ')

                xmin, ymin, xmax, ymax = list(map(int, data[:4]))
                patch_size = 64
                offset = (2 * patch_num ) * patch_size + 64
                x1 = xmin - patch_size
                x2 = xmax + patch_size
                y1 = ymin - patch_size
                y2 = ymax + patch_size
                if x1 < 0:
                    x1 = 0
                    x2 = x1 + offset
                if x2 >= w:
                    x2 = w
                    x1 = x2 - offset

                if y1 < 0:
                    y1 = 0
                    y2 = y1 + offset
                if y2 >= h:
                    y2 = h
                    y1 = y2 - offset

                cut_img = img[y1:y2, x1:x2]
                cut_limg = limg[y1:y2, x1:x2]
                # cut_mimg = mimg[dy:dy+offset, dx:dx+offset]
                # flag = int(judge(cut_img))
                # print((cut_img),(cut_limg))
                outpath = per_img.split('.')[0] + '_{}_{}_{}_{}.png'.format(y1, y2, x1, x2)
                cnt += 1
                # print(img.shape,dy,dy+offset,dx,dx+offset)
                # print(np.count_nonzero(cut_limg))
                cut_limg[cut_limg!=0] = 255
                # cut_mimg[cut_mimg>0] = 255
                # print(np.count_nonzero(cut_limg))
                cv2.imwrite(os.path.join(out_imgs_dir, 'imgs', outpath), cut_img)
                cv2.imwrite(os.path.join(out_imgs_dir, 'masks', outpath), cut_limg)
                # cv2.imwrite(os.path.join(
                #     out_imgs_dir, 'segs', outpath), cut_mimg)
        print(cnt)


if __name__ == "__main__":
    hz_root = '/nfs/DamDetection/data'
    nb_root = '/mnt/hangzhou_116_homes/DamDetection/data'
    root = hz_root
    data_source = 'dataV2'
    # root - nb_root
    result_path = os.path.join(root, data_source, 'result/Jun02_06_33_42-filter(3)')
    txt_path = os.path.join(result_path, 'box')

    # raw imgs
    raw_imgs_dir = os.path.join(root, data_source, 'image')
    # labels
    raw_limgs_dir = os.path.join(root, data_source, 'label')
    # mask
    # raw_mimgs_dir = '/home/wj/local/crack_segmentation/CrackFormer/CrackFormer-II/test_result'
    # out_path
    out_imgs_dir = os.path.join(root, 'cutDataset/overlap0.6_ts5000_slice448', data_source)
    # for path in os.listdir(out_imgs_dir):
    #     os.rmdir(os.path.join(out_imgs_dir, path))
    for root, dirs, files in os.walk(out_imgs_dir, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))

    if not os.path.exists(out_imgs_dir):
        os.makedirs(out_imgs_dir)
    if not os.path.exists(os.path.join(out_imgs_dir, 'imgs')):
        os.makedirs(os.path.join(out_imgs_dir, 'imgs'))
        os.makedirs(os.path.join(out_imgs_dir, 'masks'))
        # os.makedirs(os.path.join(out_imgs_dir, 'segs'))

    # simple_slice(
    #     raw_txt_dir=txt_path,
    #     raw_imgs_dir=raw_imgs_dir,
    #     raw_limgs_dir=raw_limgs_dir,
    #     raw_mimgs_dir=None,
    #     out_imgs_dir=out_imgs_dir,
    # )

    cut_overlap_img(
        root_path=root,
        raw_imgs_dir=raw_imgs_dir,
        raw_limgs_dir=raw_limgs_dir,
        out_imgs_dir=out_imgs_dir,
        overlap=0.6,
        slice_size=448,
        threshold=1000
    )

