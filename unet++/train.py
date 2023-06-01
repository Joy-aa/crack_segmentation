import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from d2l import torch as d2l
import tqdm
from simple_Unetpp import UnetPlusPlus
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import sys
sys.path.append("/home/wj/local/crack_segmentation")
from data_loader import ImgDataSet
import numpy as np
import os
import argparse
from pathlib import Path
from torch.autograd import Variable
import datetime

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

def train_ch13(model, train_iter, test_iter, loss, trainer, num_epochs, validation,
               devices=d2l.try_all_gpus()):
    timer, num_batches = d2l.Timer(), len(train_iter)
    # animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
    #                         legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(model, device_ids=devices).to(devices[0])
    
    loss_list = []
    valid_loss_list=[]
    epochs_list = []
    time_list = []
    for epoch in range(num_epochs):
        
        tq = tqdm.tqdm(total=(len(train_loader) * 4))
        tq.set_description(f'Epoch {epoch}')
        losses = AverageMeter()

        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(
                net, features, labels, loss.float(), trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            # if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
            #     animator.add(epoch + (i + 1) / num_batches,
            #                  (metric[0] / metric[2], metric[1] / metric[3],
            #                   None))
            losses.update(metric[0] / metric[2])
            tq.set_postfix(loss='{:.5f}'.format(losses.avg))
            tq.update(4)

        valid_metrics = validation(net, test_iter, loss)
        valid_loss = valid_metrics['valid_loss']
        # print(f'\tvalid_loss = {valid_loss:.5f}')
        print(f"epoch {epoch} --- train loss {losses.avg:.5f}--- valid loss {valid_loss:.5f}  --- cost time {timer.sum()}")
        tq.close()
        
        # # ---------保存训练数据---------------
        df = pd.DataFrame()
        loss_list.append(metric[0] / metric[2])
        valid_loss_list.append(valid_loss)
        epochs_list.append(epoch)
        time_list.append(timer.sum())
        
        df['epoch'] = epochs_list
        df['train_loss'] = loss_list
        df['valid_loss'] = valid_loss_list
        df['time'] = time_list
        today = str(datetime.date.today())
        df.to_excel(f"savefile/Unet++_crack_{today}.xlsx")
        #----------------保存模型-------------------
        if np.mod(epoch+1, 2) == 0:
            torch.save(net.state_dict(), f'checkpoints/Unet++_{epoch+1}.pth')

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(train_loader, model, criterion, optimizer, validation, args):

    print('Started training model from epoch 0')
    epoch = 0

    valid_losses = []
    for epoch in range(epoch, args.n_epoch + 1):

        adjust_learning_rate(optimizer, epoch, args.lr)

        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description(f'Epoch {epoch}')

        losses = AverageMeter()

        model.train()
        for i, (input, target) in enumerate(train_loader):
            input_var  = Variable(input).cuda()
            target_var = Variable(target).cuda()

            masks_pred = model(input_var)

            masks_probs_flat = masks_pred.view(-1)
            true_masks_flat  = target_var.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)
            losses.update(loss)
            tq.set_postfix(loss='{:.5f}'.format(losses.avg))
            tq.update(args.batch_size)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        valid_metrics = validation(model, valid_loader, criterion)
        valid_loss = valid_metrics['valid_loss']
        valid_losses.append(valid_loss)
        print(f'\tvalid_loss = {valid_loss:.5f}')
        tq.close()

        #save the model of the current epoch
        if np.mod(epoch+1, 5) == 0:
            # epoch_model_path = os.path.join(*[args.model_dir, f'model_epoch_{epoch}.pt'])
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'valid_loss': valid_loss,
                'train_loss': losses.avg
            }, f'checkpoints/Unet++_{epoch+1}.pth')

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

            losses.update(loss.item(), input_var.size(0))

    return {'valid_loss': losses.avg}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--n_epoch', default=20, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--print_freq', default=20, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--batch_size',  default=4, type=int,  help='weight decay (default: 1e-4)')
    parser.add_argument('--num_workers', default=4, type=int, help='output dataset directory')
    parser.add_argument('--data_dir',type=str, help='input dataset directory')
    # /mnt/hangzhou_116_homes/ymd/seg_dataset
    # parser.add_argument('--model_dir', type=str, help='output dataset directory')

    args = parser.parse_args()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # os.makedirs(args.model_dir, exist_ok=True)

    DIR_IMG  = os.path.join(args.data_dir, 'images')
    DIR_MASK = os.path.join(args.data_dir, 'masks')

    img_names  = [path.name for path in Path(DIR_IMG).glob('*.jpg')]
    mask_names = [path.name for path in Path(DIR_MASK).glob('*.jpg')]

    print(f'total images = {len(img_names)}')

    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(channel_means, channel_stds)])

    val_tfms = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(channel_means, channel_stds)])

    mask_tfms = transforms.Compose([transforms.ToTensor()])

    dataset = ImgDataSet(img_dir=DIR_IMG, img_fnames=img_names, img_transform=train_tfms, mask_dir=DIR_MASK, mask_fnames=mask_names, mask_transform=mask_tfms)
    train_size = int(0.85*len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)

    model = UnetPlusPlus(num_classes=1).cuda()
    # model.cuda()
    lossf = nn.BCEWithLogitsLoss()
    #选用adam优化器来训练
    # optimizer = optim.SGD(model.parameters(),lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, last_epoch=-1)
    
    #训练50轮
    # train_ch13(model, train_loader, valid_loader, lossf, optimizer, args.n_epoch, validate)

    # train(train_loader, model, lossf, optimizer, validate, args)

    validate(model, valid_loader, lossf)
    
