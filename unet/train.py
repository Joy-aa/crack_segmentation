import torch
from unet_transfer import UNet16, UNetResNet, UNet16V2
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import shutil
import sys
sys.path.append("/home/wj/pycharmProjects/crack_segmentation")
from logger import BoardLogger
from data_loader import ImgDataSet
import os
import datetime
import argparse
import tqdm
import numpy as np
import scipy.ndimage as ndimage
from build_unet import BinaryFocalLoss, dice_loss

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

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

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.5 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def find_latest_model_path(dir):
    model_paths = []
    epochs = []
    for path in Path(dir).glob('*.pt'):
        if 'epoch' not in path.stem:
            continue
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

def train(train_loader, valid_loader, model, criterion, optimizer, validation, args, logger):

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
        epoch = 0
        min_val_los = 9999

    valid_losses = []
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, para['steps'], para['gamma'])
    total_iter = 0
    # for epoch in range(0, args.n_epoch + 1):
    for epoch in range(epoch, args.n_epoch + 1):

        adjust_learning_rate(optimizer, epoch, args.lr)

        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
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
            # loss += criterion1(masks_probs_flat, true_masks_flat)
            loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), target_var.squeeze(1).float(), multiclass=False)
            losses.update(loss)
            # total_iter = 
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
        # logger.log_scalar('train/lr', optimizer.get_lr()[0], i)
        logger.log_scalar('valid/loss', valid_loss, epoch)
        print(f'\tvalid_loss = {valid_loss:.5f}')
        tq.close()

        #save the model of the current epoch
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
            input_var = Variable(input).cuda()
            target_var = Variable(target).cuda()

            output = model(input_var)
            loss = criterion(output, target_var)
            # loss += criterion1(output, target_var)
            loss += dice_loss(F.sigmoid(output.squeeze(1)), target_var.squeeze(1).float(), multiclass=False)

            losses.update(loss.item(), input_var.size(0))

    return {'valid_loss': losses.avg}

def save_check_point(state, is_best, file_name = 'checkpoint.pth.tar'):
    torch.save(state, file_name)
    if is_best:
        shutil.copy(file_name, 'model_best.pth.tar')

def calc_crack_pixel_weight(mask_dir):
    avg_w = 0.0
    n_files = 0
    for path in Path(mask_dir).glob('*.*'):
        n_files += 1
        m = ndimage.imread(path)
        ncrack = np.sum((m > 0)[:])
        w = float(ncrack)/(m.shape[0]*m.shape[1])
        avg_w = avg_w + (1-w)

    avg_w /= float(n_files)

    return avg_w / (1.0 - avg_w)


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
    # /home/wj/dataset/seg_dataset /nfs/ymd/DamCrack
    parser.add_argument('--model_dir', type=str, help='output dataset directory')
    parser.add_argument('--model_type', type=str, required=False, default='vgg16', choices=['vgg16', 'vgg16V2', 'resnet101', 'resnet34'])

    args = parser.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    TRAIN_IMG  = os.path.join(args.data_dir, 'train_image')
    TRAIN_MASK = os.path.join(args.data_dir, 'train_label')
    VALID_IMG = os.path.join(args.data_dir, 'val_image')
    VALID_MASK = os.path.join(args.data_dir, 'val_label')


    train_img_names  = [path.name for path in Path(TRAIN_IMG).glob('*.jpg')]
    train_mask_names = [path.name for path in Path(TRAIN_MASK).glob('*.bmp')]
    valid_img_names  = [path.name for path in Path(VALID_IMG).glob('*.jpg')]
    valid_mask_names = [path.name for path in Path(VALID_MASK).glob('*.bmp')]

    print(f'total train images = {len(train_img_names)}')
    print(f'total valid images = {len(valid_img_names)}')

    device = torch.device("cuda")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_gpu = torch.cuda.device_count()

    model = create_model(args.model_type)
    model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    model.to(device)

    
    long_id = '%s_%s' % (str(args.lr), datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    logger = BoardLogger(long_id)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # criterion = torch.nn.BCEWithLogitsLoss().to(device)
    criterion = BinaryFocalLoss().to(device)

    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(channel_means, channel_stds)])

    val_tfms = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(channel_means, channel_stds)])

    mask_tfms = transforms.Compose([transforms.ToTensor()])

    train_dataset = ImgDataSet(img_dir=TRAIN_IMG, img_fnames=train_img_names, img_transform=train_tfms, mask_dir=TRAIN_MASK, mask_fnames=train_mask_names, mask_transform=mask_tfms)
    valid_dataset = ImgDataSet(img_dir=VALID_IMG, img_fnames=valid_img_names, img_transform=train_tfms, mask_dir=VALID_MASK, mask_fnames=valid_mask_names, mask_transform=mask_tfms)
    

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)

    model.cuda()

    train(train_loader, valid_loader, model, criterion, optimizer, validate, args, logger)

