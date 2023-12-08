from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
from pathlib import Path
import cv2

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes
from utils import ext_transforms as et
import torchvision.transforms as transforms
from torch.utils.data import random_split
import torch.nn.functional as F
# from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/wj/local/crack_segmentation")
from data_loader import ImgDataSet
from LossFunctions import BinaryFocalLoss, dice_loss
from metric import calc_metric
from logger import BoardLogger
import datetime

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='crack',
                        choices=['voc', 'cityscapes','crack'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet50',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--epoch", type=int, default=50,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='step', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    # parser.add_argument("--step_size", type=int, default=10000)
    # parser.add_argument("--crop_val", action='store_true', default=False,
    #                     help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=4,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    # parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)
    parser.add_argument("--save_dir", type=str, default='/home/wj/local/crack_segmentation/DeepLabV3Plus/result/crackls315')

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0,1',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=42,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=100,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=1000,
                        help="epoch interval for eval (default: 100)")
    # parser.add_argument("--download", action='store_true', default=False,
    #                     help="download datasets")

    # # PASCAL VOC Options
    # parser.add_argument("--year", type=str, default='2012',
    #                     choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'crack':
        # first stage
        TRAIN_IMG  = os.path.join(opts.data_dir, 'image')

    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            # et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)
    return train_dst, val_dst

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.dice_score = 0
        self.dice_loss = 0
        self.acc = 0
        self.recall = 0
        self.precision = 0

    def update(self, loss=None, metrics=None, n=1):
        self.count += n
        if loss is not None:
            self.val = loss / n
            self.sum += loss 
            self.avg = self.sum / self.count
        if metrics is not None:
            self.dice_score += metrics['f1']
            self.precision += metrics['precision']
            self.recall += metrics['recall']

def validate(opts, model, loader, device, criterion, threshold=0.5, save_path=None):
    """Do validation and return specified samples"""
    losses = AverageMeter()
    if save_path is not None:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

    with torch.no_grad():
        idx = 1
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            targets = labels.to(device, dtype=torch.float32)

            outputs = model(images)
            loss = criterion(outputs, targets).item()
            dloss = dice_loss(outputs.squeeze(1), targets.squeeze(1), multiclass=False).item()
            metrics = calc_metric(torch.sigmoid(outputs).cpu(), targets.cpu(), mode='tensor', threshold=threshold)
            loss += dloss
            losses.update(loss=loss,metrics=metrics,n=images.size(0))
            
            if save_path is not None:
                    image = images
                    target = targets
                    pred = outputs

                    image = (denorm(image) * 255).squeeze(0).contiguous().cpu().numpy()
                    image = image.transpose(2, 1, 0).astype(np.uint8)
                    mask = torch.sigmoid(pred.squeeze(1)).contiguous().cpu().numpy()
                    label = target.squeeze(1).contiguous().cpu().numpy()

                    # print(image.shape)
                    # print(mask.shape)
                    # print(label.shape)
                    # mask = np.expand_dims(mask,axis=0)
                    # label= np.expand_dims(label,axis=0)

                    mask = (mask.transpose(2,1,0)*255).astype('uint8')
                    label = (label.transpose(2,1,0)*255).astype('uint8')
                    mask[mask>127] = 255
                    label[label>0] = 255

                    zeros = np.zeros(mask.shape)
                    mask = np.concatenate((mask,zeros,zeros),axis=-1).astype(np.uint8)
                    label = np.concatenate((zeros,zeros,label),axis=-1).astype(np.uint8)
                    
                    temp = cv2.addWeighted(label,1,mask,1,0)
                    res = cv2.addWeighted(image,0.6,temp,0.4,0)

                    cv2.imwrite(os.path.join(save_path,'test_loader/%d_test.png' % idx), res)
                    idx += 1


    num = len(loader)
    return {'loss': losses.avg, 'dice_score': losses.dice_score/num, 'precision': losses.precision/num, 'recall': losses.recall/num}


def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1-epoch/num_epochs)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_ckpt(path, epoch, model, optimizer, scheduler, best_score):
        """ save current model
        """
        torch.save({
            "epoch": epoch,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
    elif opts.dataset.lower() == 'crack':
        opts.num_classes = 1

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device("cuda")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_gpu = torch.cuda.device_count()
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    # if opts.dataset == 'voc' and not opts.crop_val:
    #     opts.val_batch_size = 1

    # first stage
    TRAIN_IMG  = os.path.join(opts.data_root, 'image')
    TRAIN_MASK = os.path.join(opts.data_root, 'label')
    train_img_names  = [path.name for path in Path(TRAIN_IMG).glob('*.jpg')]
    train_mask_names = [path.name for path in Path(TRAIN_MASK).glob('*.jpg')]
    print(f'total train images = {len(train_img_names)}')

    
    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(channel_means, channel_stds)])
    
    mask_tfms = transforms.Compose([transforms.ToTensor()])
    
    _dataset = ImgDataSet(img_dir=TRAIN_IMG, img_fnames=train_img_names, img_transform=train_tfms, mask_dir=TRAIN_MASK, mask_fnames=train_mask_names, mask_transform=mask_tfms)
    _size = int(len(_dataset) * 0.9)
    _dataset, test_dataset = random_split(_dataset, [275, 40],torch.Generator().manual_seed(42))
    # train_size = int(_size * 0.9)
    # train_dst, val_dst = random_split(_dataset, [train_size, _size - train_size],torch.Generator().manual_seed(42))
    test_loader = torch.utils.data.DataLoader(test_dataset, 1, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=4)

    # second part
    # TRAIN_IMG  = os.path.join(opts.data_root, 'imgs')
    # TRAIN_MASK = os.path.join(opts.data_root, 'masks')


    # train_img_names  = [path.name for path in Path(TRAIN_IMG).glob('*.png')]
    # train_mask_names = [path.name for path in Path(TRAIN_MASK).glob('*.png')]

    # channel_means = [0.485, 0.456, 0.406]
    # channel_stds  = [0.229, 0.224, 0.225]
    # train_tfms = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize(channel_means, channel_stds)])

    # val_tfms = transforms.Compose([transforms.ToTensor(),
    #                             transforms.Normalize(channel_means, channel_stds)])

    # mask_tfms = transforms.Compose([transforms.ToTensor()])

    # train_dataset = ImgDataSet(img_dir=TRAIN_IMG, img_fnames=train_img_names, img_transform=train_tfms, mask_dir=TRAIN_MASK, mask_fnames=train_mask_names, mask_transform=mask_tfms)
    # _size = int(len(_dataset) * 0.9)
    # _, test_dataset = random_split(_dataset, [_size, len(_dataset) - _size],torch.Generator().manual_seed(42))
    # # train_size = int(_size * 0.9)
    # # train_dst, val_dst = random_split(_dataset, [train_size, _size - train_size],torch.Generator().manual_seed(42))
    # test_loader = torch.utils.data.DataLoader(test_dataset, 1, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=4)

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=1, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    # metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)

    if opts.lr_policy == 'poly':
        total_iters = opts.epoch * int(0.9 * len(_dataset) / opts.batch_size)
        scheduler = utils.PolyLR(optimizer, total_iters, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(reduction='mean')
    # criterion = criterion.to(device)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    epoch = 0
    # cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            epoch = checkpoint["epoch"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    long_id = 'crackls_%s_%s' % (str(opts.lr), datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    logger = BoardLogger(long_id)
    vis_sample_id = np.random.randint(0, len(test_loader), opts.vis_num_samples, np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score = validate( opts=opts, model=model, loader=test_loader, device=device, criterion=criterion, threshold=0.5, save_path=opts.save_dir)
        print(val_score)
        return

    train_size = int(len(_dataset)*0.9)
    train_dataset, valid_dataset = random_split(_dataset, [265, 10])
    train_loader = torch.utils.data.DataLoader(train_dataset, opts.batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=4, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, 1, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=4)
    for epoch in range(epoch, opts.epoch+1):
    # while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        # train_size = int(len(_dataset)*0.9)
        # train_dataset, valid_dataset = random_split(_dataset, [train_size, len(_dataset) - train_size])
        # train_loader = torch.utils.data.DataLoader(train_dataset, opts.batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=4, drop_last=True)
        # valid_loader = torch.utils.data.DataLoader(valid_dataset, 1, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=4)

        tq = tqdm(total=(len(train_loader) * opts.batch_size))
        tq.set_description('Epoch %d --- Training --- :' % epoch)

        interval_loss = AverageMeter()

        for (images, labels) in train_loader:
            # cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss += dice_loss(torch.sigmoid(outputs.squeeze(1)), labels.squeeze(1).float(), multiclass=False)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss.update(np_loss)

            tq.set_postfix(loss='{:.5f}'.format(interval_loss.avg))
            tq.update(opts.batch_size)

        tq.close()

        if epoch % 5 == 0:
            save_ckpt('%s/checkpoints/%s_%s_os%d_%d.pth' % (opts.save_dir, opts.model, opts.dataset, opts.output_stride, epoch), epoch, model, optimizer, scheduler, best_score)
        # -------------------- val ------------------- #
        model.eval()
        val_score= validate(opts=opts, model=model, loader=valid_loader, device=device, criterion=criterion,)
        print(val_score)
        if val_score['dice_score'] > best_score:  # save best model
            best_score = val_score['dice_score']
            save_ckpt('%s/checkpoints/best_%s_%s_os%d.pth' % (opts.save_dir, opts.model, opts.dataset, opts.output_stride), epoch, model, optimizer, scheduler, best_score)

        logger.log_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
        logger.log_scalar('train/loss', interval_loss.avg, epoch)
        logger.log_scalar('valid/loss', val_score['loss'], epoch)
        logger.log_scalar('valid/f1', val_score['dice_score'], epoch)
        logger.log_scalar('valid/precision', val_score['precision'], epoch)
        logger.log_scalar('valid/recall', val_score['recall'], epoch)

        scheduler.step()
    val_score = validate( opts=opts, model=model, loader=test_loader, device=device, criterion=criterion, threshold=0.5, save_path=opts.save_dir)

if __name__ == '__main__':
    main()
