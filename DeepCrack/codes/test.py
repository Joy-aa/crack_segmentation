# from data.dataset import readIndex, dataReadPip, loadedDataset
from model.deepcrack import DeepCrack
from trainer import DeepCrackTrainer
import cv2
from tqdm import tqdm
import numpy as np
import torch
import os
from pathlib import Path
import torchvision.transforms as transforms
import sys
sys.path.append("/home/wj/local/crack_segmentation")
from metric import *
from data_loader import ImgDataSet
from PIL import Image
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def test(test_data_path='/home/wj/local/crack_segmentation/images',
         save_path='deepcrack_results/',
         pretrained_model='checkpoints/DeepCrack_CT260_FT1/epoch(9)_acc(0.23493-0.98289).pth', ):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # test_pipline = dataReadPip(transforms=None)

    # test_list = readIndex(test_data_path)

    # test_dataset = loadedDataset(test_list, preprocess=test_pipline)

    DIR_IMG  = os.path.join(test_data_path, 'image')
    DIR_MASK = os.path.join(test_data_path, 'new_label')

    img_paths  = [path for path in Path(DIR_IMG).glob('*.*')]
    # mask_names = [path.name for path in Path(DIR_MASK).glob('*.*')]

    print(f'total images = {len(img_paths)}')

    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]

    # val_tfms = transforms.Compose([transforms.ToTensor(),transforms.Normalize(channel_means, channel_stds)])
    val_tfms = transforms.Compose([transforms.ToTensor()])

    mask_tfms = transforms.Compose([transforms.ToTensor()])

    # dataset = ImgDataSet(img_dir=DIR_IMG, img_fnames=img_names, img_transform=val_tfms, mask_dir=DIR_MASK, mask_fnames=mask_names, mask_transform=mask_tfms)

    # test_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
    #                                           shuffle=False, num_workers=1, drop_last=False)

    # -------------------- build trainer --------------------- #

    device = torch.device("cuda")
    num_gpu = torch.cuda.device_count()

    model = DeepCrack()

    model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    model.to(device)

    trainer = DeepCrackTrainer(model).to(device)

    model.load_state_dict(trainer.saver.load(pretrained_model, multi_gpu=True))

    model.eval()

    input_size = (448, 448)
    cof = 1
    w, h = int(cof * input_size[0]), int(cof * input_size[1])

    pred_list= []
    gt_list=[]

    with torch.no_grad():
        for path in tqdm(img_paths):
            print(path)
            img = cv2.imread(str(path), 1)
            mask_path = os.path.join(DIR_MASK, path.stem+'.bmp')
            lab = cv2.imread(mask_path, 0)
            # print(img.shape)
            # print(lab.shape)
        
            # img = img.unsqueeze(0)
            # img = np.transpose(img, (1, 2, 0))
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
                    img_pat = img[i1:i2, j1:j2]
                    mask_pat = lab[i1:i2, j1:j2]
                    if i2-i1 != h or j2-j1 != w:
                        img_pat = cv2.resize(img_pat, (w, h), cv2.INTER_AREA)
                        mask_pat = cv2.resize(mask_pat, (w, h), cv2.INTER_AREA)
                    test_data = val_tfms(Image.fromarray(img_pat))
                    test_target = mask_tfms(Image.fromarray(mask_pat))
                    test_data, test_target = Variable(test_data.unsqueeze(0)).cuda(), Variable(test_target.unsqueeze(0)).cuda()
                    # test_data, test_target = img_pat.type(torch.cuda.FloatTensor).to(device), mask_pat.type(torch.cuda.FloatTensor).to(device)
                    test_pred = trainer.val_op(test_data, test_target)
                    test_pred = torch.sigmoid(test_pred[0].squeeze()).data.cpu().numpy()
                    if i2-i1 != h or j2-j1 != w:
                        test_pred = cv2.resize(test_pred, (j2-j1, i2-i1), cv2.INTER_AREA)
                    img_1[i1:i2, j1:j2] = test_pred
            # img_1[img_1 > 1] = 1
            pred_list.append(img_1)
            gt_list.append(lab)
            save_name = os.path.join(save_path, f'{path.stem}.jpg')
            # print(save_name)
            cv2.imwrite(filename=save_name, img=(img_1 * 255).astype(np.uint8))
    metric = calc_metric(pred_list, gt_list, mode='list', threshold=0.5)
    with open('result.txt', 'a', encoding='utf-8') as fout:
            print(metric)
            line =  "accuracy:{:.5f} | precision:{:.5f} | recall:{:.5f} | f1:{:.5f} " \
                .format(metric['accuracy'], metric['precision'], metric['recall'], metric['f1']) + '\n'
            fout.write(line)


if __name__ == '__main__':
    test()
