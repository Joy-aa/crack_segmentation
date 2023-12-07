import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
import numpy as np
# from d2l import torch as d2l
from tqdm import tqdm
from unet.network.unet_transfer import UNet16, UNetResNet, UNet16V2
# import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
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

def create_model(type ='vgg16'):
    if type == 'vgg16':
        print('create vgg16 model')
        model = UNet16(pretrained=True)
    elif type == 'vgg16V2':
        print('create vgg16V2 model')
        model = UNet16V2(pretrained=True)
    elif type == 'resnet101':
        encoder_depth = 101
        num_classes = 1
        print('create resnet101 model')
        model = UNetResNet(encoder_depth=encoder_depth, num_classes=num_classes, pretrained=True)
    elif type == 'resnet34':
        encoder_depth = 34
        num_classes = 1
        print('create resnet34 model')
        model = UNetResNet(encoder_depth=encoder_depth, num_classes=num_classes, pretrained=True)
    else:
        assert False
    model.eval()
    return model

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
    lr = lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(train_loader, valid_loader, model, criterion, optimizer, validation, args):

    # latest_model_path = find_latest_model_path(args.model_dir)
    # latest_model_path = os.path.join(*[args.model_dir, 'model_start.pt'])
    latest_model_path = "/mnt/hangzhou_116_homes/wj/model/unet2/model_best.pt"
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

        #load the min loss so far
        best_state = torch.load(latest_model_path)
        min_val_los = best_state['valid_loss']

        print(f'Restored model at epoch {epoch}. Min validation loss so far is : {min_val_los}')
        epoch += 1
        print(f'Started training model from epoch {epoch}')
    else:
        print('Started training model from epoch 0')
        epoch = 0
        min_val_los = 9999

    valid_losses = []
    total_iter = 0
    for epoch in range(epoch, args.n_epoch + 1):

        adjust_learning_rate(optimizer, epoch, args.lr)

        tq = tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description(f'Epoch {epoch}')

        losses = AverageMeter()
        report_interval = 100
        model.train()
        for i, (input, target) in enumerate(train_loader):
            input_var  = Variable(input).cuda()
            target_var = Variable(target).cuda()

            masks_pred = model(input_var)

            masks_probs_flat = masks_pred.view(-1)
            true_masks_flat  = target_var.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)
            loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), target_var.squeeze(1).float(), multiclass=False)
            losses.update(loss)

            if (i+total_iter) % report_interval == 0:
                logger.log_scalar('train/lr', optimizer.param_groups[0]['lr'], i+total_iter)
                logger.log_scalar('train/loss', losses.avg, i+total_iter)

            tq.set_postfix(loss='{:.5f}'.format(losses.avg))
            tq.update(args.batch_size)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_iter += len(train_loader)
        valid_metrics = validation(model, valid_loader, criterion)
        valid_loss = valid_metrics['valid_loss']
        valid_losses.append(valid_loss)
        logger.log_scalar('valid/loss', valid_loss, epoch)
        print(f'\tvalid_loss = {valid_loss:.5f}')
        tq.close()

        #save the model of the current epoch
        if np.mod(epoch+1, 5) == 0:
            epoch_model_path = os.path.join(*[args.model_dir, f'model_epoch_{epoch}.pt'])
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'valid_loss': valid_loss,
                'train_loss': losses.avg
            }, epoch_model_path)
        
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

            output = model(input_var)
            loss = criterion(output, target_var)
            loss += dice_loss(F.sigmoid(output.squeeze(1)), target_var.squeeze(1).float(), multiclass=False)
            losses.update(loss.item(), input_var.size(0))

    return {'valid_loss': losses.avg}

def predict(test_loader, model, latest_model_path):
    model.eval()
    metrics=[]
    pred_list = []
    gt_list = []
    bar = tqdm(total=len(test_loader))
    with torch.no_grad():
        for idx, (img, lab) in enumerate(test_loader, 1):
            val_data  = Variable(img).cuda()
            pred = model(val_data)
            # print(lab.shape)
            # print(pred.shape)
            pred = torch.sigmoid(pred.squeeze(1).contiguous().cpu()).numpy()
            lab = lab.squeeze(1).numpy()
            # print(lab.shape)
            # print(pred.shape)
            pred_list.append(pred)
            gt_list.append(lab)
            # Image.fromarray(lab.astype('uint8')).save('results/%d_target.png' % idx)
            # Image.fromarray(pred.astype('uint8')).save('results/%d_pred.png' % idx)
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
                fout.write(latest_model_path+'\n')
                for i in range(1, 10): 
                    line =  "threshold:{:d} | accuracy:{:.5f} | precision:{:.5f} | recall:{:.5f} | f1:{:.5f} " \
                        .format(i, metrics[i-1]['accuracy'],  metrics[i-1]['precision'],  metrics[i-1]['recall'],  metrics[i-1]['f1']) + '\n'
                    fout.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--n_epoch', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--print_freq', default=20, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--weight_decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--batch_size',  default=16, type=int,  help='weight decay (default: 1e-4)')
    parser.add_argument('--num_workers', default=4, type=int, help='output dataset directory')
    parser.add_argument('--data_dir',type=str, help='input dataset directory')
    # /home/wj/dataset/seg_dataset /nfs/wj/DamCrack
    parser.add_argument('--model_dir', type=str, help='output dataset directory')
    parser.add_argument('--model_type', type=str, required=False, default='vgg16', choices=['vgg16', 'vgg16V2', 'resnet101', 'resnet34'])


    args = parser.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    TRAIN_IMG  = os.path.join(args.data_dir, 'imgs')
    TRAIN_MASK = os.path.join(args.data_dir, 'masks')


    train_img_names  = [path.name for path in Path(TRAIN_IMG).glob('*.png')]
    train_mask_names = [path.name for path in Path(TRAIN_MASK).glob('*.png')]
    # train_img_names = train_img_names[:len(train_img_names)*0.1]
    # train_mask_names = train_mask_names[:len(train_mask_names)*0.1]
    print(f'total train images = {len(train_img_names)}')

    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(channel_means, channel_stds)])

    val_tfms = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(channel_means, channel_stds)])

    mask_tfms = transforms.Compose([transforms.ToTensor()])

    train_dataset = ImgDataSet(img_dir=TRAIN_IMG, img_fnames=train_img_names, img_transform=train_tfms, mask_dir=TRAIN_MASK, mask_fnames=train_mask_names, mask_transform=mask_tfms)
    _dataset, test_dataset = random_split(train_dataset, [0.9, 0.1],torch.Generator().manual_seed(42))
    train_dataset, valid_dataset = random_split(_dataset, [0.9, 0.1],torch.Generator().manual_seed(42))
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=4)
    val_loader = torch.utils.data.DataLoader(valid_dataset, 1, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, 1, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=4)


    long_id = '%s_%s' % (str(args.lr), datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    logger = BoardLogger(long_id)

    device = torch.device("cuda")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_gpu = torch.cuda.device_count()

    model = create_model(args.model_type)
    model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    model.to(device)
    #选用adam优化器来训练
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    criterion = BinaryFocalLoss().to(device)

    train(train_loader, val_loader, model, criterion, optimizer, validate, args)

    # latest_model_path = '/home/wj/local/crack_segmentation/unet++/checkpoints/Unet++_20.pth'
    # state = torch.load(latest_model_path)
    # model.load_state_dict(state['model'])
    # # weights = state['model']
    # # weights_dict = {}
    # # for k, v in weights.items():
    # #     new_k = k.replace('module.', '') if 'module' in k else k
    # #     weights_dict[new_k] = v
    # # model.load_state_dict(weights_dict)
    # predict(test_loader, model, latest_model_path)
    
