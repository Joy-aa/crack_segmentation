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
from segtool.data_loader import ImgDataSet, CrackDataset, CrackDataSetWithEdge
import os
import datetime
import argparse
import tqdm
import numpy as np
import scipy.ndimage as ndimage
from network.build_unet import BinaryFocalLoss, dice_loss
from loss import loss
from segtool.metric import calc_metric
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0,1])) 
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

def create_model(num_classes = 1, type ='vgg16', is_deconv=False):
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
        model = UNetGate(num_classes=num_classes, pretrained=True, is_deconv=is_deconv)
    else:
        assert False
    model.eval()
    return model

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 10))
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

def f_score(inputs, target):
    n, c, *_ = inputs.size() #torch.Size([n, 2, 64, 64])
    nt, *_  = target.size() #torch.Size([n, 64, 64])
    #n h c w -> n h w c -> n h*w c
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(nt, -1) #torch.Size([n, h * w])
    
    #--------------------------------------------#
    #   计算f1_score
    #--------------------------------------------#
    
    #agrmax(dim = -1)
    preds = torch.gt(temp_inputs[...,1], temp_inputs[...,0]).int().flatten() #torch.Size([n, h * w])
    
    
    label = temp_target.flatten()
    
    Positive_correct = torch.sum(torch.logical_and(label == 1, preds == 1))
    
    recall = (Positive_correct)  / (label.sum() + 1)
    precision = (Positive_correct)  / (preds.sum() + 1)
    """ 
    计算IOU
    """
    
    return 1, torch.mean(recall).cpu(), torch.mean(precision).cpu()

def train(train_loader, test_loader, model, criterion, criterion_val, optimizer, validation, args, logger):

    # latest_model_path = find_latest_model_path(args.model_dir)
    # latest_model_path = os.path.join(*[args.model_dir, 'model_start.pt'])
    latest_model_path = args.snapshot
    best_model_path = os.path.join(*[args.model_dir, 'model_best.pt'])

    if latest_model_path is not None:
        state = torch.load(latest_model_path)
        # epoch = state['epoch']
        epoch = 0
        model.load_state_dict(state['model'])
        # best_state = torch.load(latest_model_path)
        # min_val_los = best_state['valid_loss']
        min_val_los = -1

        print(f'Restored model at epoch {epoch}. Max validation metric so far is : {min_val_los}')
        epoch += 1
        print(f'Started training model from epoch {epoch}')
    else:
        print('Started training model from epoch 0')
        epoch = 1
        min_val_los = -1
        print(f'Max validation metric so far is : {min_val_los}')

    total_iter = 0
    for epoch in range(epoch, args.n_epoch + 1):

        adjust_learning_rate(optimizer, epoch, args.lr)

        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size), ncols = 120)
        tq.set_description(f'Epoch {epoch}')

        losses = AverageMeter()
        seg_loss = AverageMeter()
        edge_loss = AverageMeter()
        att_loss = AverageMeter()
        dual_loss = AverageMeter()
        dice_loss = AverageMeter()
        model.train()
        for i, (input, target, edge) in enumerate(train_loader):
            batch_pixel_size = input.size(0) * input.size(2) * input.size(3)

            input_var  = Variable(input).to(device)
            target_var = Variable(target).to(device)
            edge_var = Variable(edge).cuda()

            masks_pred = model(input_var)

            loss_dict = criterion(masks_pred, (target_var, edge_var))

            main_loss = sum(loss_dict.values()).mean()
            if(torch.isnan(main_loss)):
                print(loss_dict)
                with open(os.path.join('./result_dir', 'error.txt'), 'a+', encoding='utf-8') as fout:
                    d = datetime.datetime.today()
                    datetime.datetime.strftime(d,'%Y-%m-%d %H-%M-%S')
                    fout.write(str(d)+'\t'+loss_dict+'\n')
                exit(0)
            
            log_main_loss = main_loss.clone().detach_()

            losses.update(log_main_loss.item(), batch_pixel_size)
            if 'seg_loss' in loss_dict:
                seg_loss.update(loss_dict['seg_loss'], batch_pixel_size)
            if 'edge_loss' in loss_dict:
                edge_loss.update(loss_dict['edge_loss'], batch_pixel_size)
            if 'att_loss' in loss_dict:
                att_loss.update(loss_dict['att_loss'], batch_pixel_size)
            if 'dual_loss' in loss_dict:
                dual_loss.update(loss_dict['dual_loss'], batch_pixel_size)
            if 'dice_loss' in loss_dict:
                dice_loss.update(loss_dict['dice_loss'], batch_pixel_size)


            # total_iter = 
            if (i+total_iter) % args.print_freq == 0:
                logger.log_scalar('train/lr', optimizer.param_groups[0]['lr'], i+total_iter)
                logger.log_scalar('train/loss', losses.avg, i+total_iter)
                if 'seg_loss' in loss_dict:
                    logger.log_scalar('train/seg_loss', seg_loss.avg, i+total_iter)
                if 'edge_loss' in loss_dict:
                    logger.log_scalar('train/edge_loss', edge_loss.avg, i+total_iter)
                if 'att_loss' in loss_dict:
                    logger.log_scalar('train/att_loss', att_loss.avg, i+total_iter)
                if 'dual_loss' in loss_dict:
                    logger.log_scalar('train/dual_loss', dual_loss.avg, i+total_iter)
                if 'dice_loss' in loss_dict:
                    logger.log_scalar('train/dice_loss', dice_loss.avg, i+total_iter)
            tq.set_postfix(loss='{:.5f}'.format(losses.avg))
            tq.update(args.batch_size)
            # compute gradient and do SGD step
            optimizer.zero_grad()
            main_loss.backward()
            optimizer.step()

        total_iter += len(train_loader)
        valid_metrics = validation(model, test_loader, criterion_val, args.r0, args.r1)
        logger.log_scalar('valid/precision', valid_metrics['precision'], epoch)
        logger.log_scalar('valid/recall', valid_metrics['recall'], epoch)
        logger.log_scalar('valid/f1', valid_metrics['f1'], epoch)
        valid_metric = valid_metrics['f1']
        print(f'\nvalid_f1 = {valid_metric:.5f}')
        tq.close()

        # save the model of the current epoch
        epoch_model_path = os.path.join(*[args.model_dir, f'model_epoch_{epoch}.pt'])
        if(epoch % 10 == 0):
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'valid_loss': valid_metric,
                'train_loss': losses.avg
            }, epoch_model_path)

        if valid_metric > min_val_los:
            min_val_los = valid_metric

            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'valid_loss': valid_metric,
                'train_loss': losses.avg
            }, best_model_path)

def validate(model, val_loader, criterion, r0 = True, r1 = 10):
    losses = AverageMeter()
    Fscore = AverageMeter()
    Recall = AverageMeter()
    Precision = AverageMeter()
    model.eval()
    pbar = tqdm.tqdm(total=len(val_loader),desc='val',postfix=dict,mininterval=0.3)
    with torch.no_grad():
        for i, (input, target, edge) in enumerate(val_loader):
            # batch_pixel_size = input.size(0) * input.size(2) * input.size(3)

            input_var = Variable(input).to(device)
            target_var = Variable(target).to(device)
            edge_var = Variable(edge).to(device)

            pred, edge_out = model(input_var)
            _, recall, precision = f_score(pred, target_var)
            Recall.update(recall, input.size(0))
            Precision.update(precision, input.size(0))
            if precision != 0 or recall != 0:
                Fscore.update(2 * recall * precision / (recall + precision), input.size(0))
            pbar.set_postfix(**{ 'f1': Fscore.avg,'recall': Recall.avg, 'precision':Precision.avg,})
            pbar.update(input.size(0))
            # loss_dict = criterion((pred, edge_out), (target_var, edge_var))
            # losses.update(sum(loss_dict.values()).item(), batch_pixel_size)
            # loss = calc_loss(pred, target_var, criterion, r0, r1)

            # losses.update(loss.item())

    return {'recall': Recall.avg,
            'precision':Precision.avg,
            'f1': Fscore.avg}

def predict(test_loader, model, latest_model_path, save_dir = './result/test_loader', vis_sample_id = None):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    pred_list = []
    gt_list = []
    denorm = Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    bar = tqdm.tqdm(total=len(test_loader), ncols = 100)
    with torch.no_grad():
        for idx, (img, lab, edge) in enumerate(test_loader, 1):
            val_data = Variable(img).to(device)
            pred, edge_out = model(val_data)

            image = (denorm(img) * 255).squeeze(0).contiguous().cpu().numpy()
            image = image.transpose(2, 1, 0).astype(np.uint8)
            n, c, h, w = pred.size()
            temp_inputs = torch.softmax(pred.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
            preds = torch.gt(temp_inputs[...,1], temp_inputs[...,0]).int().squeeze(-1)
            mask = preds.squeeze(0).view(h,w).contiguous().cpu().numpy()
            mask = np.expand_dims(mask, axis=0)
            label = lab.contiguous().cpu().numpy()

            pred_list.append(mask)
            gt_list.append(label)

            if idx in vis_sample_id:
                mask = (mask.transpose(2, 1, 0)*255).astype('uint8')
                mask[mask > 127] = 255
                mask[mask < 255] = 0
                # crackInfos = measure(mask=mask[0])

                label = (label.transpose(2, 1, 0)*255).astype('uint8')
                label[label>0] = 255
                
                zeros = np.zeros(mask.shape)
                mask = np.concatenate((mask,zeros,zeros),axis=-1).astype(np.uint8)
                label = np.concatenate((zeros,zeros,label),axis=-1).astype(np.uint8)
                
                temp = cv2.addWeighted(label,1,mask,1,0)
                res = cv2.addWeighted(image,0.6,temp,0.4,0)

                # edge_pred = torch.sigmoid(edge_out.squeeze(0)).contiguous().cpu().numpy().transpose(2, 1, 0)
                edge_pred =edge_out.squeeze(0).contiguous().cpu().numpy().transpose(2, 1, 0)

                # # 开始统计
                # # 将数据拉平为一维数组
                # flat_data = edge_pred.flatten()
                # # 定义分组边界，这里按照0-1均分成10组
                # bins = np.linspace(0, 1, 11)
                # # 使用histogram函数统计各区间内的数据个数
                # hist, bin_edges = np.histogram(flat_data, bins=bins)
                # # 打印各区间的数据个数
                # sum = 0
                # print()
                # for i in range(0,len(hist)):
                #     sum += hist[i]
                #     print(f"区间[{bin_edges[i]:.1f}, {bin_edges[i + 1]:.1f}) 的数据个数: {hist[i]} ，至今总个数：{sum}")

                edge_mask = np.where(edge_pred > 0.001, edge_pred*100, zeros)
                edge_mask[edge_mask > 1] = 1
                edge_mask = np.concatenate((zeros,zeros,edge_mask * 255),axis=-1).astype(np.uint8)
                res_edge = cv2.addWeighted(image,0.6,edge_mask,0.4,0)

                cv2.imwrite(os.path.join(save_dir,'%d_test.png' % idx), res)
                cv2.imwrite(os.path.join(save_dir,'%d_test_edge.png' % idx), res_edge)
                
                # with open(os.path.join(save_dir,'%d_test.txt' % idx), 'a', encoding='utf-8') as fout:
                #     for crack in crackInfos:
                #         line =  "box:[{:d},{:d},{:d},{:d}] | length:{:.5f} | avg_width:{:.5f} | max_width:{:.5f} " \
                #             .format( crack['box'][0], crack['box'][1], crack['box'][2], crack['box'][3], crack['length'],  crack['avg_width'],  crack['max_width']) + '\n'
                #         fout.write(line)
            bar.update(1)
    bar.close

    metric = calc_metric(pred_list, gt_list, mode='list', threshold=0.5, max_value=1)
    print(metric)
    d = datetime.datetime.today()
    datetime.datetime.strftime(d,'%Y-%m-%d %H-%M-%S')
    os.makedirs('./result_dir', exist_ok=True)
    with open(os.path.join('./result_dir', str(d)+'.txt'), 'a', encoding='utf-8') as fout:
                fout.write(str(latest_model_path)+'\n')
                # for i in range(1, 10): 
                line =  "threshold:0.5 | accuracy:{:.5f} | precision:{:.5f} | recall:{:.5f} | f1:{:.5f} | miou:{:.5f} " \
                    .format(metric['accuracy'],  metric['precision'],  metric['recall'],  metric['f1'],  metric['miou']) + '\n'
                fout.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--n_epoch', default=20, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--print_freq', default=100, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--weight_decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--batch_size',  default=4, type=int,  help='weight decay (default: 1e-4)')
    parser.add_argument('--num_workers', default=4, type=int, help='output dataset directory')
    parser.add_argument('--num_classes', default=2, type=int, help='output dataset directory')
    parser.add_argument('--data_dir',type=str, default='/mnt/hangzhou_116_homes/DamDetection/data/cutDataset/overlap0.6_ts1000_slice224/total/', help='input dataset directory')
    # /nfs/wj/DamCrack /nfs/wj/192_255_segmentation /mnt/hangzhou_116_homes/DamDetection/data/cutDataset/overlap0.6_ts1000_slice224/total/ /mnt/hangzhou_116_homes/zek/crackseg9k/
    parser.add_argument('--model_dir', type=str, default='/home/wj/local/crack_segmentation/unet/checkpoints/crackseg9k/loss', help='output dataset directory')
    parser.add_argument('--model_type', type=str, required=False, default='gate', choices=['vgg16', 'vgg16V2', 'vgg16V3', 'gate'])
    parser.add_argument('--snapshot', type=str, default="/home/wj/local/crack_segmentation/unet/checkpoints/crackseg9k/loss/model_best.pt")
    parser.add_argument('--joint_edgeseg_loss', action='store_true', default=False,help='joint loss')
    parser.add_argument('--r0',  action='store_true', default=True,  help='seg loss dice weight')
    parser.add_argument('--r1',  default=10, type=int,  help='seg loss bce weight')
    parser.add_argument('--r2', default=20, type=int, help='edge loss weight')
    parser.add_argument('--att_th', type=float, default=0.8,help='Attention loss weight for joint loss')
    parser.add_argument('--normal',  action='store_true', default=False,  help='seg loss dice weight')
    parser.add_argument("--deconv", action='store_true', default=False)

    args = parser.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    train_Dataset = CrackDataSetWithEdge('train',[args.data_dir])
    test_dataset = CrackDataSetWithEdge('val',[args.data_dir]) 

    train_loader = torch.utils.data.DataLoader(train_Dataset, args.batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, 1, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)

    model = create_model(args.num_classes, args.model_type, args.deconv)
    model = torch.nn.DataParallel(model, device_ids=[0,1])
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # criterion = torch.nn.BCEWithLogitsLoss().to(device)
    # # criterion = BinaryFocalLoss()
    criterion, criterion_val = loss.get_loss(args)

    long_id = 'unetgate_%s_%s' % (str(args.lr), datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    logger = BoardLogger(long_id)
    train(train_loader, test_loader, model, criterion, criterion_val, optimizer, validate, args, logger)

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

