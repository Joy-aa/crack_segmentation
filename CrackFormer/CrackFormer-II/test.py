from utils.utils import *
from utils.newValidator import *
from utils.Crackloader import *
from nets.crackformerII import crackformer
from nets.SDDNet import SDDNet
from nets.STRNet import STRNet
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import cv2 as cv
import sys
sys.path.append("/home/wj/local/crack_segmentation")
from metric import *
import bisect

def Test(valid_img_dir, valid_lab_dir, valid_result_dir, valid_log_dir, best_model_dir, model, input_size = (512, 1024, 2048)):
    
    validator = Validator(model, valid_log_dir, best_model_dir)
    paths = [path for path in Path(valid_img_dir).glob('*.*')]

    metrics = {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
    }
    for path in tqdm(paths):
        print(path)
        filepath = os.path.join('/home/wj/dataset/Jun02_06_33_42/box', path.stem+'.txt')
        boxes = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for data in f.readlines():
                box = data.split(' ')[:-1]
                boxes.append(box)

        pred_list = []
        gt_list = []
        img_0 = cv.imread(str(path), 1)
        img_0 = np.asarray(img_0)
        if len(img_0.shape) != 3:
            print(f'incorrect image shape: {path.name}{img_0.shape}')
            continue

        img_0 = img_0[:,:,:3]
        img_height, img_width, *img_channels = img_0.shape

        if valid_lab_dir != '':
            mask_path = os.path.join(valid_lab_dir, path.stem+'.png')
            lab = cv.imread(mask_path, 0)
        else:
            lab = np.zeros(img_height, img_width)

        img_1 = np.zeros((img_height, img_width))

        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            img_pat = img_0[y1:y2, x1:x2]
            gt_pat = lab[y1:y2, x1:x2]
            ori_shape = gt_pat.shape
            img_pat = cv.resize(img_pat, (128, 128), cv.INTER_AREA)
            gt_pat = cv.resize(gt_pat, (128, 128), cv.INTER_AREA)
            prob_map_full = validator.validate(img_pat)
            gt_list.append(gt_pat)
            # print(gt_pat.shape)
            # print(prob_map_full.shape)
            pred_list.append(prob_map_full)
            prob_map_full = cv.resize(prob_map_full, (ori_shape[1], ori_shape[0]), cv.INTER_AREA)
            img_1[y1:y2, x1:x2] = prob_map_full

        if args.out_pred_dir != '':
            img_1[img_1 > 0.1] = 1
            img_1[img_1 <= 0.1] = 0
            cv.imwrite(filename=os.path.join(args.out_pred_dir, f'{path.stem}.jpg'), img=(img_1 * 255).astype(np.uint8))

        metric = calc_metric(pred_list, gt_list, mode='list', threshold=0.1)
        print(metric)
        metrics['accuracy'] += metric['accuracy'] / len(paths)
        metrics['precision'] += metric['precision'] / len(paths)
        metrics['recall'] += metric['recall'] / len(paths)
        metrics['f1'] += metric['f1'] / len(paths)

    print(metrics)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir',type=str, default='/home/wj/dataset/crack', help='input dataset directory')
    parser.add_argument('--model_path', type=str, default='/home/wj/local/crack_segmentation/CrackFormer/CrackFormer-II/model/crackformer_epoch(14).pth', help='trained model path')
    parser.add_argument('--model_type', type=str, default='crackformer', choices=['crackformer', 'SDDNet', 'STRNet'])
    parser.add_argument('--out_pred_dir', type=str, default='./test_result', required=False,  help='prediction output dir')
    args = parser.parse_args()

    if args.out_pred_dir != '':
        os.makedirs(args.out_pred_dir, exist_ok=True)
        for path in Path(args.out_pred_dir).glob('*.*'):
            os.remove(str(path))

    if args.model_type == 'crackformer':
        model = crackformer()
    elif args.model_type  == 'SDDNet':
        model = SDDNet(3, 1)
    elif args.model_type  == 'STRNet':
        model = STRNet(3, 1)
    else:
        print('undefind model name pattern')
        exit()
    model.load_state_dict(torch.load(args.model_path))
    model.cuda()
    DIR_IMG  = os.path.join(args.img_dir, 'image')
    DIR_MASK = os.path.join(args.img_dir, 'new_label')
    valid_log_dir = "./log/" + args.model_type + '/'
    best_model_dir = "./model/" + args.model_type + "/"
    # image_format = "jpg"

    Test(DIR_IMG, DIR_MASK, args.out_pred_dir, valid_log_dir, best_model_dir, model)