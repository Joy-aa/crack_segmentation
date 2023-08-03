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


def simple_slice(raw_txt_dir: str, raw_imgs_dir: str, raw_limgs_dir: str, raw_mimgs_dir: str, out_imgs_dir: str, overlap=0.0, patch_num=1):
    '''
     根据标注图片切割目录下的所有图片，并进行二分类
    '''
    # 非裂缝图片数量
    cnt = 0
    # 裂缝图片数量

    for per_img in tqdm(os.listdir(raw_imgs_dir)):
        # limg_name = per_img.split('.')[0]+'_mask.png'
        limg_path = os.path.join(raw_limgs_dir, per_img.split('.')[0]+'.png')
        mimg_path = os.path.join(raw_mimgs_dir, per_img.split('.')[0]+'.jpg')
        img_path = os.path.join(raw_imgs_dir, per_img)
        txt_path = os.path.join(raw_txt_dir, per_img.split('.')[0]+'.txt')
        if os.path.isdir(img_path) or per_img.split('.')[-1] == 'txt':
            continue
        # 图像读取
        img = cv2.imread(img_path)
        # limg 是标注图片
        limg = cv2.imread(limg_path, 0)
        mimg = cv2.imread(mimg_path, 0)
        h, w, *_ = img.shape
        # 切割
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                data = line.split(' ')

                xmin, ymin, xmax, ymax = list(map(int, data[:4]))
                patch_size = 64
                offset = (2 * patch_num ) * patch_size + 64
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

                # print(dy, dy+offset, dx, dx+offset)
                cut_img = img[dy:dy+offset, dx:dx+offset]
                cut_limg = limg[dy:dy+offset, dx:dx+offset]
                cut_mimg = mimg[dy:dy+offset, dx:dx+offset]
                # flag = int(judge(cut_img))
                # print((cut_img),(cut_limg))
                outpath = os.path.join('img{}.png'.format(cnt))
                cnt += 1
                # print(img.shape,dy,dy+offset,dx,dx+offset)
                # print(np.count_nonzero(cut_limg))
                cut_limg[cut_limg!=0] = 255
                cut_mimg[cut_mimg>0] = 255
                # print(np.count_nonzero(cut_limg))
                cv2.imwrite(os.path.join(
                    out_imgs_dir, 'imgs', outpath), cut_img)
                cv2.imwrite(os.path.join(
                    out_imgs_dir, 'masks', outpath), cut_limg)
                cv2.imwrite(os.path.join(
                    out_imgs_dir, 'segs', outpath), cut_mimg)


if __name__ == "__main__":
    hz_root = '/mnt/hangzhou_116_homes'
    nb_root = '/mnt/hangzhou_116_homes'
    root_path = hz_root+'/DamDetection/data/result-stride_0.7/Jun02_06_33_42/'
    # root_path = hz_root+'/DamDetection/data'
    txt_path = root_path + '/box'

    # raw imgs
    raw_imgs_dir = os.path.join(hz_root, 'DamDetection/data/image')
    # labels
    raw_limgs_dir = os.path.join(hz_root, 'DamDetection/data/new_label')
    # mask
    raw_mimgs_dir = '/home/wj/local/crack_segmentation/CrackFormer/CrackFormer-II/test_result'
    # out_path
    out_imgs_dir = os.path.join(hz_root, 'wj/192_255_segmentation')

    if not os.path.exists(out_imgs_dir):
        os.makedirs(out_imgs_dir)
    if not os.path.exists(os.path.join(out_imgs_dir, 'imgs')):
        os.makedirs(os.path.join(out_imgs_dir, 'imgs'))
        os.makedirs(os.path.join(out_imgs_dir, 'masks'))
        os.makedirs(os.path.join(out_imgs_dir, 'segs'))

    simple_slice(
        raw_txt_dir=txt_path,
        raw_imgs_dir=raw_imgs_dir,
        raw_limgs_dir=raw_limgs_dir,
        raw_mimgs_dir=raw_mimgs_dir,
        out_imgs_dir=out_imgs_dir,
        overlap=0.4,
    )
