import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
import numpy as np
# from d2l import torch as d2l
from tqdm import tqdm
import cv2
from simple_Unetpp import UnetPlusPlus
# import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms.functional import normalize
from torch.utils.data import DataLoader, random_split
import sys
sys.path.append("/home/wj/local/crack_segmentation")
from logger import BoardLogger
from data_loader import ImgDataSet
from LossFunctions import BinaryFocalLoss, dice_loss
from metric import calc_metric
import numpy as np
import os
import argparse
from pathlib import Path
from torch.autograd import Variable
import datetime
import torch.nn.functional as F

class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 15))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def find_latest_model_path(dir):
    model_paths = []
    epochs = []
    for path in Path(dir).glob('*.pth'):
        # if 'epoch' not in path.stem:
        #     continue
        model_paths.append(path)
        parts = path.stem.split('_')
        epoch = int(parts[-1])
        epochs.append(epoch)
        
    if len(epochs) > 0:
        epochs = np.array(epochs)
        max_idx = np.argmax(epochs)
        return model_paths[max_idx]
    else:
        return None

def calc_loss(masks_pred, target_var, criterion, r=1):
    masks_probs_flat = masks_pred.view(-1)
    true_masks_flat  = target_var.view(-1)
    
    loss = r * criterion(masks_probs_flat, true_masks_flat)
    loss += dice_loss(torch.sigmoid(masks_pred.squeeze(1)), target_var.squeeze(1).float(), multiclass=False)
    return loss

def train(dataset, model, criterion, optimizer, validation, args, logger):

    # latest_model_path = find_latest_model_path(args.model_dir)
    # latest_model_path = os.path.join(*[args.model_dir, 'model_start.pt'])
    latest_model_path = None
    best_model_path = os.path.join(*[args.model_dir, 'model_best.pt'])

    if latest_model_path is not None:
        state = torch.load(latest_model_path)
        epoch = state['epoch']
        model.load_state_dict(state['model'])
        # weights = state['model']
        # weights_dict = {}
        # for k, v in weights.items():
        #     new_k = k.replace('module.', '') if 'module' in k else k
        #     weights_dict[new_k] = v
        # model.load_state_dict(weights_dict)

        #if latest model path does exist, best_model_path should exists as well
        assert Path(best_model_path).exists() == True, f'best model path {best_model_path} does not exist'
        #load the min loss so far
        best_state = torch.load(latest_model_path)
        min_val_los = best_state['valid_loss']

        print(f'Restored model at epoch {epoch}. Min validation loss so far is : {min_val_los}')
        epoch += 1
        print(f'Started training model from epoch {epoch}')
    else:
        print('Started training model from epoch 0')
        epoch = 1
        min_val_los = 9999

    valid_losses = []
    total_iter = 0
    # train_size = int(len(dataset)*0.9)
    train_dataset, valid_dataset = random_split(dataset, [265, 10])
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, 1, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)
    for epoch in range(epoch, args.n_epoch + 1):

        # train_size = int(len(dataset)*0.9)
        # train_dataset, valid_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
        # train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)
        # valid_loader = torch.utils.data.DataLoader(valid_dataset, 1, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)

        adjust_learning_rate(optimizer, epoch, args.lr)

        tq = tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description(f'Epoch {epoch}')

        losses = AverageMeter()
        model.train()
        for i, (input, target) in enumerate(train_loader):
            input_var  = Variable(input).cuda()
            target_var = Variable(target).cuda()

            masks_pred = model(input_var)[1]

            loss = calc_loss(masks_pred, target_var, criterion, 10)
            losses.update(loss)

            # if (i+total_iter) % args.print_freq == 0:
            #     logger.log_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
            #     logger.log_scalar('train/loss', losses.avg, epoch)

            tq.set_postfix(loss='{:.5f}'.format(losses.avg))
            tq.update(args.batch_size)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.log_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
        logger.log_scalar('train/loss', losses.avg, epoch)
        total_iter += len(train_loader)
        valid_metrics = validation(model, valid_loader, criterion)
        valid_loss = valid_metrics['valid_loss']
        valid_losses.append(valid_loss)
        logger.log_scalar('valid/loss', valid_loss, epoch)
        print(f'\tvalid_loss = {valid_loss:.5f}')
        tq.close()

        #save the model of the current epoch
        if np.mod(epoch, 5) == 0:
            # epoch_model_path = os.path.join(*[args.model_dir, f'model_epoch_{epoch}.pt'])
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'valid_loss': valid_loss,
                'train_loss': losses.avg
            }, f'{args.model_dir}/Unet++_step2_{epoch}.pth')

        if valid_loss < min_val_los:
            min_val_los = valid_loss

            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'valid_loss': valid_loss,
                'train_loss': losses.avg
            }, best_model_path)

def validate(model, val_loader, criterion):
    losses = AverageMeter()
    model.eval()
    with torch.no_grad():

        for i, (input, target) in enumerate(val_loader):
            # print(input.shape)
            # print(target.shape)
            input_var = Variable(input).cuda()
            target_var = Variable(target).cuda()

            output = model(input_var)[1]
            loss = calc_loss(output, target_var, criterion, 10)
            losses.update(loss.item())

    return {'valid_loss': losses.avg}

def predict(test_loader, model, latest_model_path, save_dir = './result/test_loader'):
    model.eval()
    metrics=[]
    pred_list = []
    gt_list = []
    denorm = Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    bar = tqdm(total=len(test_loader))
    with torch.no_grad():
        for idx, (img, lab) in enumerate(test_loader, 1):
            val_data  = Variable(img).cuda()
            pred = model(val_data)[1]
            pred = torch.sigmoid(pred.squeeze(1).contiguous().cpu()).numpy()
            lab = lab.squeeze(1).numpy()
            pred_list.append(pred)
            gt_list.append(lab)
            
            mask = (pred.transpose(2, 1, 0)*255).astype('uint8')
            label = (lab.transpose(2, 1, 0)*255).astype('uint8')
            mask[mask>127] = 255
            label[label>0] = 255
            
            zeros = np.zeros(mask.shape)
            mask = np.concatenate((mask,zeros,zeros),axis=-1).astype(np.uint8)
            label = np.concatenate((zeros,zeros,label),axis=-1).astype(np.uint8)
            
            image = (denorm(img) * 255).squeeze(0).contiguous().cpu().numpy()
            image = image.transpose(2, 1, 0).astype(np.uint8)
            temp = cv2.addWeighted(label,1,mask,1,0)
            res = cv2.addWeighted(image,0.6,temp,0.4,0)

            cv2.imwrite(os.path.join(save_dir,'%d_test.png' % idx), res)

            bar.update(1)
    bar.close

    for i in range(1, 10):
                threshold = i / 10
                metric = calc_metric(pred_list, gt_list, mode='list', threshold=threshold)
                print(metric)
                if len(metrics) < i:
                    metrics.append(metric)
                else:
                    metrics[i-1]['accuracy'] += metric['accuracy']
                    metrics[i-1]['precision'] += metric['precision']
                    metrics[i-1]['recall'] += metric['recall']
                    metrics[i-1]['f1'] += metric['f1']
    print(metrics)
    d = datetime.datetime.today()
    datetime.datetime.strftime(d,'%Y-%m-%d %H-%M-%S')
    os.makedirs('./result_dir', exist_ok=True)
    with open(os.path.join('./result_dir', str(d)+'.txt'), 'a', encoding='utf-8') as fout:
                fout.write(str(latest_model_path)+'\n')
                for i in range(1, 10): 
                    line =  "threshold:{:d} | accuracy:{:.5f} | precision:{:.5f} | recall:{:.5f} | f1:{:.5f} " \
                        .format(i, metrics[i-1]['accuracy'],  metrics[i-1]['precision'],  metrics[i-1]['recall'],  metrics[i-1]['f1']) + '\n'
                    fout.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--n_epoch', default=50, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--print_freq', default=100, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--batch_size',  default=4, type=int,  help='weight decay (default: 1e-4)')
    parser.add_argument('--num_workers', default=4, type=int, help='output dataset directory')
    parser.add_argument('--data_dir',type=str, help='input dataset directory')
    # /mnt/hangzhou_116_homes/wj/192_255_segmentation/
    parser.add_argument('--model_dir', type=str, help='output dataset directory')

    args = parser.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
# 第一阶段训练
    # TRAIN_IMG  = os.path.join(args.data_dir, 'image')
    # TRAIN_MASK = os.path.join(args.data_dir, 'label')
    # train_img_names  = [path.name for path in Path(TRAIN_IMG).glob('*.jpg')]
    # train_mask_names = [path.name for path in Path(TRAIN_MASK).glob('*.png')]
    # print(f'total train images = {len(train_img_names)}')

    # channel_means = [0.485, 0.456, 0.406]
    # channel_stds  = [0.229, 0.224, 0.225]
    # train_tfms = transforms.Compose([transforms.ToTensor(),
    #                                  transforms.Normalize(channel_means, channel_stds)])
    # val_tfms = transforms.Compose([transforms.ToTensor(),
    #                                transforms.Normalize(channel_means, channel_stds)])
    # mask_tfms = transforms.Compose([transforms.ToTensor()])

    # train_dataset = ImgDataSet(img_dir=TRAIN_IMG, img_fnames=train_img_names, img_transform=train_tfms, mask_dir=TRAIN_MASK, mask_fnames=train_mask_names, mask_transform=mask_tfms)

# 第二阶段训练
    TRAIN_IMG  = os.path.join(args.data_dir, 'image')
    TRAIN_MASK = os.path.join(args.data_dir, 'label')
    print(args.data_dir)
    train_img_names  = [path.name for path in Path(TRAIN_IMG).glob('*.jpg')]
    train_mask_names = [path.name for path in Path(TRAIN_MASK).glob('*.jpg')]
    print(f'total train images = {len(train_img_names)}')

    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(channel_means, channel_stds)])

    val_tfms = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(channel_means, channel_stds)])

    mask_tfms = transforms.Compose([transforms.ToTensor()])

    train_dataset = ImgDataSet(img_dir=TRAIN_IMG, img_fnames=train_img_names, img_transform=train_tfms, mask_dir=TRAIN_MASK, mask_fnames=train_mask_names, mask_transform=mask_tfms)
    _dataset, test_dataset = random_split(train_dataset, [275, 40],torch.Generator().manual_seed(42))
    # train_dataset, valid_dataset = random_split(_dataset, [0.9, 0.1],torch.Generator().manual_seed(42))
    # train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)
    # val_loader = torch.utils.data.DataLoader(valid_dataset, 1, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, 1, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=4)


    long_id = 'unet++_crackls_%s_%s' % (str(args.lr), datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    logger = BoardLogger(long_id)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device = torch.device("cuda")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_gpu = torch.cuda.device_count()
    model = UnetPlusPlus(num_classes=1, deep_supervision=True)
    model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    #选用adam优化器来训练
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # train(_dataset, model, criterion, optimizer, validate, args, logger)

    latest_model_path = find_latest_model_path(args.model_dir)
    state = torch.load(latest_model_path)
    model.load_state_dict(state['model'])
    # weights = state['model']
    # weights_dict = {}
    # for k, v in weights.items():
    #     new_k = k.replace('module.', '') if 'module' in k else k
    #     weights_dict[new_k] = v
    # model.load_state_dict(weights_dict)
    predict(test_loader, model, latest_model_path, '/home/wj/local/crack_segmentation/unet++/result/crackls315_out3/test_loader')
    
