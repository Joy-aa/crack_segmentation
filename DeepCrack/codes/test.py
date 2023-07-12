# from data.dataset import readIndex, dataReadPip, loadedDataset
import sys
sys.path.append("/home/wj/local/crack_segmentation")
from model.deepcrack import DeepCrack
from model.deepcrackv2 import DeepCrackV2
from trainer import DeepCrackTrainer
import cv2
from tqdm import tqdm
import numpy as np
import torch
import os
from pathlib import Path
# from config import Config as cfg
import torchvision.transforms as transforms
from metric import *
from data_loader import ImgDataSet
from PIL import Image
from torch.autograd import Variable
import bisect

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def test(test_data_path='/mnt/ningbo_nfs_36/wj/data/',
         save_path='deepcrack_results/',
<<<<<<< HEAD
         pretrained_model='checkpoints/DeepCrack_CT260_FT1/epoch(73)_acc(0.32035-0.99489).pth', ):
=======
         pretrained_model='checkpoints/DeepCrack_CT260_FT1/epoch(68)_acc(0.16684-0.99606).pth', ):
>>>>>>> e9f39ef9011b2c7ec67e08d5ba7393a433da6809
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # test_pipline = dataReadPip(transforms=None)

    # test_list = readIndex(test_data_path)

    # test_dataset = loadedDataset(test_list, preprocess=test_pipline)

    DIR_IMG  = os.path.join(test_data_path, 'image')
    DIR_MASK = os.path.join(test_data_path, 'new_label')

    paths  = [path for path in Path(DIR_IMG).glob('*.*')]
    # mask_names = [path.name for path in Path(DIR_MASK).glob('*.*')]

    print(f'total images = {len(paths)}')

    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]

    # val_tfms = transforms.Compose([transforms.ToTensor(),transforms.Normalize(channel_means, channel_stds)])
    val_tfms = transforms.Compose([transforms.ToTensor()])

    mask_tfms = transforms.Compose([transforms.ToTensor()])

    # dataset = ImgDataSet(img_dir=DIR_IMG, img_fnames=img_names, img_transform=val_tfms, mask_dir=DIR_MASK, mask_fnames=mask_names, mask_transform=mask_tfms)

    # test_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
    #                                           shuffle=False, num_workers=1, drop_last=False)

    # -------------------- build trainer --------------------- #

    # device = torch.device("cuda")
    # num_gpu = torch.cuda.device_count()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model = DeepCrack()
    # model = DeepCrackV2()

    # model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    model.to(device)

    trainer = DeepCrackTrainer(model)
    # checkpoint = trainer.saver.load(pretrained_model, multi_gpu=True)
    # weights = checkpoint['model']
    # weights_dict = {}
    # for k, v in weights.items():
    #         new_k = k.replace('module.', '') if 'module' in k else k
    #         weights_dict[new_k] = v
    # model.load_state_dict(weights_dict)

    model.load_state_dict(trainer.saver.load(pretrained_model, multi_gpu=False))
    model.cuda()

    model.eval()

    input_size = (448, 448)
    cof = 1
    w, h = int(cof * input_size[0]), int(cof * input_size[1])
    offset = 32

    # metrics = {
    #         'accuracy': 0,
    #         'precision': 0,
    #         'recall': 0,
    #         'f1': 0,
    # }
    metrics=[]
    with torch.no_grad():
        for path in tqdm(paths):
            pred_list= []
            gt_list=[]
            print(path)
            img = cv2.imread(str(path), 1)
            mask_path = os.path.join(DIR_MASK, path.stem+'.png')
            lab = cv2.imread(mask_path, 0)
            img_height, img_width, *img_channels = img.shape
            img_1 = np.zeros((img_height, img_width))
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
                    img_pat = img[i1:i2 + offset, j1:j2 + offset]
                    mask_pat = lab[i1:i2 + offset, j1:j2 + offset]
                    ori_shape = mask_pat.shape
                    if mask_pat.shape != (h+offset, w+offset):
                        img_pat = cv2.resize(img_pat, (w+offset, h+offset), cv2.INTER_AREA)
                        mask_pat = cv2.resize(mask_pat, (w+offset, h+offset), cv2.INTER_AREA)
                        test_data = val_tfms(Image.fromarray(img_pat))
                        test_target = mask_tfms(Image.fromarray(mask_pat))
                        test_data, test_target = Variable(test_data.unsqueeze(0)).cuda(), Variable(test_target.unsqueeze(0)).cuda()
                        test_pred = trainer.val_op(test_data, test_target)
                        test_pred = torch.sigmoid(test_pred[0].squeeze()).data.cpu().numpy()
                        pred_list.append(test_pred)
                        gt_list.append(mask_pat)
                        test_pred = cv2.resize(test_pred, (ori_shape[1], ori_shape[0]), cv2.INTER_AREA)
                    else:
                        test_data = val_tfms(Image.fromarray(img_pat))
                        test_target = mask_tfms(Image.fromarray(mask_pat))
                        test_data, test_target = Variable(test_data.unsqueeze(0)).cuda(), Variable(test_target.unsqueeze(0)).cuda()
                        test_pred = trainer.val_op(test_data, test_target)
                        test_pred = torch.sigmoid(test_pred[0].squeeze()).data.cpu().numpy()
                        pred_list.append(test_pred)
                        gt_list.append(mask_pat)

                    img_1[i1:i2 + offset, j1:j2 + offset] +=  test_pred
            img_1[img_1 > 1] = 1
            save_name = os.path.join(save_path, f'{path.stem}.jpg')
            cv2.imwrite(filename=save_name, img=(img_1 * 255).astype(np.uint8))

                    
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
                fout.write(pretrained_model+'\n')
                for i in range(1, 10): 
                    line =  "threshold:{:d} | accuracy:{:.5f} | precision:{:.5f} | recall:{:.5f} | f1:{:.5f} " \
                        .format(i, metrics[i-1]['accuracy'],  metrics[i-1]['precision'],  metrics[i-1]['recall'],  metrics[i-1]['f1']) + '\n'
                    fout.write(line)


if __name__ == '__main__':
    test()
