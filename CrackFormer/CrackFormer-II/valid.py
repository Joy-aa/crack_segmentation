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

def Test(valid_img_dir, valid_lab_dir, valid_result_dir, valid_log_dir, best_model_dir, model, input_size = (512, 1024, 2048), threshold = 0.1):
    
    validator = Validator(model, valid_log_dir, best_model_dir)
    cof = 1
    w, h = int(cof * input_size[0]), int(cof * input_size[1])
    img_paths = [path for path in Path(valid_img_dir).glob('*.*')]
    # pred_list= []
    # gt_list=[]

    metrics = {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
    }
    for path in tqdm(img_paths):
        pred_list=[]
        gt_list = []
        print(str(path))
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
            lab = np.zeros((img_height, img_width))


        img_1 = np.zeros((img_height, img_width))

        # 切割处理
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
                    if i2-i1 != h or j2-j1 != w:
                        img_pat = cv.resize(img_pat, (w, h), cv.INTER_AREA)
                        mask_pat = cv2.resize(mask_pat, (w, h), cv2.INTER_AREA)
                        prob_map_full = validator.validate(img_pat)
                        pred_list.append(prob_map_full)
                        gt_list.append(mask_pat)
                        prob_map_full = cv.resize(prob_map_full, (j2-j1, i2-i1), cv.INTER_AREA)
                    else:
                        prob_map_full = validator.validate(img_pat)
                        pred_list.append(prob_map_full)
                        gt_list.append(mask_pat)
                    img_1[i1:i2, j1:j2] += prob_map_full
        if valid_lab_dir != '':
            metric = calc_metric(pred_list, gt_list, mode='list', threshold=threshold)
            metrics['accuracy'] += metric['accuracy'] / len(img_paths)
            metrics['precision'] += metric['precision'] / len(img_paths)
            metrics['recall'] += metric['recall'] / len(img_paths)
            metrics['f1'] += metric['f1'] / len(img_paths)
            print(metric)
        # img_1[img_1 > 1] = 1
        img_1[img_1 > threshold] = 1
        img_1[img_1 <= threshold] = 0
        pred_mask = (img_1 * 255).astype(np.uint8)
        # img_1[img_1 > 0.1] = 1
        # img_1[img_1 <= 0.1] = 0
        cv.imwrite(filename=os.path.join(valid_result_dir, f'{path.stem}.jpg'), img=pred_mask)
    if valid_lab_dir != '':
        with open('result.txt', 'a', encoding='utf-8') as fout:
                print(metrics)
                line =  "accuracy:{:.5f} | precision:{:.5f} | recall:{:.5f} | f1:{:.5f} " \
                    .format(metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']) + '\n'
                fout.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir',type=str, default='/mnt/hangzhou_116_homes/DamDetection/data', help='input dataset directory')
    parser.add_argument('--model_path', type=str, default='/home/wj/local/crack_segmentation/CrackFormer/CrackFormer-II/model/epoch(5)_acc(0.30-0.98).pth', help='trained model path')
    parser.add_argument('--model_type', type=str, default='crackformer', choices=['crackformer', 'SDDNet', 'STRNet'])
    parser.add_argument('--out_pred_dir', type=str, default='./test_result', required=False,  help='prediction output dir')
    parser.add_argument('--type', type=str, default='out' , choices=['out', 'metric'])
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
    if args.type == 'out':
        DIR_IMG = os.path.join(args.img_dir, 'image')
        DIR_GT = ''
    elif args.type  == 'metric':
        DIR_IMG = os.path.join(args.img_dir, 'image')
        DIR_GT = os.path.join(args.img_dir, 'new_label')
    else:
        print('undefind test pattern')
        exit()
    # DIR_IMG  = os.path.join(args.img_dir, 'image')
    # DIR_MASK = os.path.join(args.img_dir, 'new_label')
    valid_log_dir = "./log/" + args.model_type + '/'
    best_model_dir = "./model/" + args.model_type + "/"
    # image_format = "jpg"

    Test(DIR_IMG, DIR_GT, args.out_pred_dir, valid_log_dir, best_model_dir, model)