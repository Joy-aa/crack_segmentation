import torch
from network.unet_transfer import UNet16
from network.unet_network import UNet16V3, UNet16V2
from network.unet_gate import UNetGate
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import normalize
import shutil
import sys
sys.path.append("/home/wj/local/crack_segmentation")
from segtool.logger import BoardLogger
from segtool.measure import measure
from segtool.data_loader import ImgDataSet
import os
import datetime
import argparse
import tqdm
import numpy as np
import scipy.ndimage as ndimage
from unet.network.build_unet import BinaryFocalLoss, dice_loss
from segtool.metric import calc_metric
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0, 1])) 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count())

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

def create_model(type ='vgg16', is_deconv=False):
    if type == 'vgg16':
        print('create vgg16 model')
        model = UNet16(pretrained=True, is_deconv=is_deconv)
    elif type == 'vgg16V2':
        print('create vgg16V2 model')
        model = UNet16V2(pretrained=True, is_deconv=is_deconv)
    elif type == 'vgg16V3':
        print('create vgg16V3 model')
        model = UNet16V3(pretrained=True, is_deconv=is_deconv)
    elif type == 'gate':
        print('create gateconv model')
        model = UNetGate(pretrained=True, is_deconv=is_deconv)
    else:
        assert False
    model.eval()
    return model

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.5 ** (epoch // 10))
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

def calc_loss(masks_pred, target_var, r=1):
    masks_probs_flat = masks_pred.view(-1)
    true_masks_flat  = target_var.view(-1)
    
    loss = criterion(masks_probs_flat, true_masks_flat)
    # loss = r * criterion(masks_probs_flat, true_masks_flat)
    # loss += dice_loss(torch.sigmoid(masks_pred.squeeze(1)), target_var.squeeze(1).float(), multiclass=False)
    return loss

def train(dataset, model, criterion, optimizer, validation, args, logger):

    latest_model_path = find_latest_model_path(args.model_dir)
    # latest_model_path = os.path.join(*[args.model_dir, 'model_start.pt'])
    latest_model_path = args.snapshot
    best_model_path = os.path.join(*[args.model_dir, 'model_best.pt'])

    if latest_model_path is not None:
        state = torch.load(latest_model_path)
        # epoch = state['epoch']
        epoch = 0
        model.load_state_dict(state['model'])
        # weights = state['model']
        # weights_dict = {}
        # for k, v in weights.items():
        #     new_k = k.replace('module.', '') if 'module' in k else k
        #     weights_dict[new_k] = v
        # model.load_state_dict(weights_dict)

        #if latest model path does exist, best_model_path should exists as well
        # assert Path(best_model_path).exists() == True, f'best model path {best_model_path} does not exist'
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
    # train_dataset, valid_dataset = random_split(dataset, [265, 10])
    # train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)
    # valid_loader = torch.utils.data.DataLoader(valid_dataset, 1, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)
    for epoch in range(epoch, args.n_epoch + 1):
        train_size = int(len(dataset)*0.9)
        train_dataset, valid_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, 1, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)

        adjust_learning_rate(optimizer, epoch, args.lr)

        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size), ncols=150)
        tq.set_description(f'Epoch {epoch}')

        losses = AverageMeter()
        model.train()
        for i, (input, target) in enumerate(train_loader):
            input_var  = Variable(input).to(device)
            target_var = Variable(target).to(device)

            masks_pred = model(input_var)

            loss = calc_loss(masks_pred, target_var, 10)
            losses.update(loss)
            # total_iter = 
            if (i+total_iter) % args.print_freq == 0:
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
        epoch_model_path = os.path.join(*[args.model_dir, f'model_epoch_{epoch}.pt'])
        if(epoch % 10 == 0):
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
            input_var = Variable(input).to(device)
            target_var = Variable(target).to(device)

            pred = model(input_var)
            loss = calc_loss(pred, target_var, 10)

            losses.update(loss.item())

    return {'valid_loss': losses.avg}

def predict(test_loader, model, latest_model_path, save_dir = './result/test_loader', vis_sample_id = None):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    metrics=[]
    pred_list = []
    gt_list = []
    denorm = Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    bar = tqdm.tqdm(total=len(test_loader), ncols=100)
    with torch.no_grad():
        for idx, (img, lab) in enumerate(test_loader, 1):
            # val_data  = Variable(img).cuda()
            val_data = Variable(img).to(device)
            pred = model(val_data)

            image = (denorm(img) * 255).squeeze(0).contiguous().cpu().numpy()
            image = image.transpose(2, 1, 0).astype(np.uint8)
            mask = torch.sigmoid(pred.squeeze(0)).contiguous().cpu().numpy()
            label = lab.squeeze(0).contiguous().cpu().numpy()

            pred_list.append(mask)
            gt_list.append(label)

            if idx in vis_sample_id:
                mask = (mask.transpose(2, 1, 0)*255).astype('uint8')
                mask[mask > 127] = 255
                mask[mask < 255] = 0
                # crackInfos = measure(mask=mask)

                label = (label.transpose(2, 1, 0)*255).astype('uint8')
                label[label>0] = 255
                
                zeros = np.zeros(mask.shape)
                mask = np.concatenate((mask,zeros,zeros),axis=-1).astype(np.uint8)
                label = np.concatenate((zeros,zeros,label),axis=-1).astype(np.uint8)
                # print(label.shape)
                
                temp = cv2.addWeighted(label,1,mask,1,0)
                res = cv2.addWeighted(image,0.6,temp,0.4,0)

                cv2.imwrite(os.path.join(save_dir,'%d_test.png' % idx), res)
                
                # with open(os.path.join(save_dir,'%d_test.txt' % idx), 'a', encoding='utf-8') as fout:
                #     for crack in crackInfos:
                #         line =  "box:[{:d},{:d},{:d},{:d}] | length:{:.5f} | avg_width:{:.5f} | max_width:{:.5f} " \
                #             .format( crack['box'][0], crack['box'][1], crack['box'][2], crack['box'][3], crack['length'],  crack['avg_width'],  crack['max_width']) + '\n'
                #         fout.write(line)
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
                    line =  "threshold:{:d} | accuracy:{:.5f} | precision:{:.5f} | recall:{:.5f} | f1:{:.5f} | miou:{:.5f}" \
                        .format(i, metrics[i-1]['accuracy'],  metrics[i-1]['precision'],  metrics[i-1]['recall'],  metrics[i-1]['f1'], metrics[i-1]['miou']) + '\n'
                    fout.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--n_epoch', default=30, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--print_freq', default=100, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--weight_decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--batch_size',  default=4, type=int,  help='weight decay (default: 1e-4)')
    parser.add_argument('--num_workers', default=8, type=int, help='output dataset directory')
    parser.add_argument('--data_dir',type=str, default='/mnt/hangzhou_116_homes/DamDetection/data/cutDataset/overlap0.6_ts1000_slice224/dataV2', help='input dataset directory')
    # /home/wj/dataset/seg_dataset /nfs/wj/DamCrack /nfs/wj/192_255_segmentation
    parser.add_argument('--model_dir', type=str, default='/home/wj/local/crack_segmentation/unet/checkpoints/cutDatasetV2', help='output dataset directory')
    parser.add_argument('--snapshot', type=str, default=None, help='pretrained model')
    parser.add_argument('--model_type', type=str, required=False, default='vgg16', choices=['vgg16', 'vgg16V2', 'vgg16V3', 'gate'])
    parser.add_argument("--deconv", action='store_true', default=False)

    args = parser.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    # 第一阶段训练
    # TRAIN_IMG  = os.path.join(args.data_dir, 'images')
    # TRAIN_MASK = os.path.join(args.data_dir, 'labels')
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
    TRAIN_IMG  = os.path.join(args.data_dir, 'imgs')
    TRAIN_MASK = os.path.join(args.data_dir, 'masks')


    train_img_names  = [path.name for path in Path(TRAIN_IMG).glob('*.png')]
    train_mask_names = [path.name for path in Path(TRAIN_MASK).glob('*.png')]
    print(f'total train images = {len(train_img_names)}')

    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(channel_means, channel_stds)])

    val_tfms = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(channel_means, channel_stds)])

    mask_tfms = transforms.Compose([transforms.ToTensor()])

    train_dataset = ImgDataSet(img_dir=TRAIN_IMG, img_fnames=train_img_names, img_transform=train_tfms, mask_dir=TRAIN_MASK, mask_fnames=train_mask_names, mask_transform=mask_tfms)
    # train_dataset = CrackDataSet(img_dir=TRAIN_IMG, img_fnames=train_img_names, img_transform=train_tfms, mask_dir=TRAIN_MASK, mask_fnames=train_mask_names, mask_transform=mask_tfms)
    train_size = int(len(train_dataset)*0.7)
    _dataset, test_dataset = random_split(train_dataset, [train_size, len(train_dataset) - train_size],torch.Generator().manual_seed(42))
    # train_dataset, valid_dataset = random_split(_dataset, [0.9, 0.1],torch.Generator().manual_seed(42))
    # train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=4)
    # val_loader = torch.utils.data.DataLoader(valid_dataset, 1, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, 1, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=4)


    # device = torch.device("cuda")
    # print(device)
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # num_gpu = torch.cuda.device_count()

    model = create_model(args.model_type, args.deconv)
    model = torch.nn.DataParallel(model, device_ids=[0,1])
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    # criterion = BinaryFocalLoss()

    # long_id = '%s_%s' % (str(args.lr), datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    # logger = BoardLogger(long_id)
    # train(_dataset, model, criterion, optimizer, validate, args, logger)

    np.random.seed(0)
    vis_sample_id = np.random.randint(0, len(test_loader), 50, np.int32)  # sample idxs for visualization

    latest_model_path = os.path.join(args.model_dir, 'model_best.pt')
    state = torch.load(latest_model_path)
    epoch = state['epoch']
    model.load_state_dict(state['model'])
    # weights = state['model']
    # weights_dict = {}
    # for k, v in weights.items():
    #     new_k = k.replace('module.', '') if 'module' in k else k
    #     weights_dict[new_k] = v
    # model.load_state_dict(weights_dict)
    predict(test_loader, model, latest_model_path, save_dir=os.path.join(args.model_dir, 'test_visual'), vis_sample_id=vis_sample_id)

