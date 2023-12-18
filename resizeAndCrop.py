import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


if __name__ == "__main__":
    hz_root = '/nfs/DamDetection/data/pretrainDataset/'
    nb_root = '/mnt/hangzhou_116_homes/DamDetection/data'
    root = hz_root
    data_source = 'crackforestdataset'
    # root - nb_root
    result_path = os.path.join(root, data_source, 'result/Jun02_06_33_42-filter(3)')
    txt_path = os.path.join(result_path, 'box')

    # raw imgs
    raw_imgs_dir = os.path.join(root, data_source, 'image')
    # labels
    raw_limgs_dir = os.path.join(root, data_source, 'gt')
    img_paths  = [path for path in Path(raw_limgs_dir).glob('*.png')]
    # out_path
    out_imgs_dir = os.path.join(root, 'total')
    # for path in os.listdir(out_imgs_dir):
    #     os.rmdir(os.path.join(out_imgs_dir, path))
    # for root, dirs, files in os.walk(out_imgs_dir, topdown=False):
    #     for name in files:
    #         os.remove(os.path.join(root, name))
    #     for name in dirs:
    #         os.rmdir(os.path.join(root, name))

    if not os.path.exists(out_imgs_dir):
        os.makedirs(out_imgs_dir)
    if not os.path.exists(os.path.join(out_imgs_dir, 'imgs')):
        os.makedirs(os.path.join(out_imgs_dir, 'imgs'))
        os.makedirs(os.path.join(out_imgs_dir, 'masks'))
    for label_path in tqdm(img_paths):
        path = os.path.join(raw_imgs_dir, label_path.stem.split('.')[0]+'.jpg')

        img = cv2.imread(str(path))
        label = cv2.imread(str(label_path), 0)
        # label[label == 38 ] = 0
        # label[label == 75 ] = 255

        # resize
        img = cv2.resize(img,  (448, 448), cv2.INTER_AREA)
        label = cv2.resize(label, (448, 448), cv2.INTER_AREA)
        label[label <= 127] = 0
        label[label > 127] = 255

        # crop
        outpath = label_path.stem.split('.')[0]+'.png'

        cv2.imwrite(os.path.join(out_imgs_dir, 'imgs', outpath), img)
        cv2.imwrite(os.path.join(out_imgs_dir, 'masks', outpath), label)
