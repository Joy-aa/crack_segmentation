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
from data_loader import ImgDataSet
from metric import *
import bisect
from torch.utils.data import DataLoader, random_split

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
    parser.add_argument('--img_dir',type=str, default='/mnt/nfs/wj/192_255_segmentation', help='input dataset directory')
    parser.add_argument('--model_path', type=str, default='/home/wj/local/crack_segmentation/CrackFormer/CrackFormer-II/model/crackformer_epoch(36).pth', help='trained model path')
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
    TRAIN_IMG  = os.path.join(args.img_dir, 'imgs')
    TRAIN_MASK = os.path.join(args.img_dir, 'masks')
    train_img_names  = [path.name for path in Path(TRAIN_IMG).glob('*.png')]
    train_mask_names = [path.name for path in Path(TRAIN_MASK).glob('*.png')]
    
    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose([transforms.ToTensor()])
                                    #  transforms.Normalize(channel_means, channel_stds)])
    val_tfms = transforms.Compose([transforms.ToTensor()])
                                #    transforms.Normalize(channel_means, channel_stds)])
    mask_tfms = transforms.Compose([transforms.ToTensor()])
    train_dataset = ImgDataSet(img_dir=TRAIN_IMG, img_fnames=train_img_names, img_transform=train_tfms, mask_dir=TRAIN_MASK, mask_fnames=train_mask_names, mask_transform=mask_tfms)
    _dataset, test_dataset = random_split(train_dataset, [0.9, 0.1],torch.Generator().manual_seed(42))
    test_loader = torch.utils.data.DataLoader(test_dataset, 1, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=4)
    print(f'total test images = {len(test_loader)}')

    valid_log_dir = "./log/" + args.model_type + '/'
    best_model_dir = "./model/" + args.model_type + "/"
    validator = Validator(model, valid_log_dir, best_model_dir)

    metrics=[]
    pred_list = []
    gt_list = []
    bar = tqdm(total=len(test_loader))
    with torch.no_grad():
        for idx, (img, lab) in enumerate(test_loader, 1):
            val_data  = Variable(img).cuda()
            _,_,_,_,_, pred = model(val_data)
            pred_list.append(torch.sigmoid(pred.contiguous().cpu()).numpy())
            gt_list.append(lab.numpy())
            bar.update(1)
    bar.close
    for i in range(1, 10):
            threshold = i / 10
            metric = calc_metric(pred_list, gt_list, mode='list', threshold=threshold)
            print(metric)
            # metric['accuracy'] = metric['accuracy'] / len(test_loader)
            # metric['precision'] = metric['precision'] / len(test_loader)
            # metric['recall'] = metric['recall'] / len(test_loader)
            # metric['f1'] = metric['f1'] / len(test_loader)
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
            fout.write(args.model_path+'\n')
            for i in range(1, 10): 
                        line =  "threshold:{:d} | accuracy:{:.5f} | precision:{:.5f} | recall:{:.5f} | f1:{:.5f} " \
                            .format(i, metrics[i-1]['accuracy'],  metrics[i-1]['precision'],  metrics[i-1]['recall'],  metrics[i-1]['f1']) + '\n'
                        fout.write(line)

