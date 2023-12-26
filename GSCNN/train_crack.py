import torch
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
import os
import datetime
import argparse
import tqdm
import numpy as np
import scipy.ndimage as ndimage
from segtool.LossFunctions import BinaryFocalLoss, dice_loss
from segtool.metric import calc_metric
import cv2
from config import cfg

import network
import optimizer
import loss
from datasets import crack
# from datasets.crack import CrackDataSet

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
    lr = lr * (0.1 ** (epoch // 20))
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
    
    loss = r * criterion(masks_probs_flat, true_masks_flat)
    loss += dice_loss(torch.sigmoid(masks_pred.squeeze(1)), target_var.squeeze(1).float(), multiclass=False)
    return loss

def train(dataset, model, criterion, optimizer, validation, args, logger, criterion_val):

    # model.train()

    # latest_model_path = find_latest_model_path(args.model_dir)
    # latest_model_path = os.path.join(*[args.model_dir, 'model_start.pt'])
    latest_model_path = args.snapshot
    best_model_path = os.path.join(*[args.model_dir, 'model_best.pt'])

    if latest_model_path is not None:
        state = torch.load(latest_model_path)
        # epoch = state['epoch']
        epoch = 0
        # model.load_state_dict(state['state_dict'])
        weights = state['state_dict']
        weights_dict = {}
        for k, v in weights.items():
            if k == 'module.criterion.seg_loss.nll_loss.weight':
                # print(k)
                continue
            weights_dict[k] = v
            # new_k = k.replace('module.', '') if 'module' in k else k
            # weights_dict[new_k] = v
        model.load_state_dict(weights_dict)

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

    # train_size = int(len(dataset)*0.9)
    train_dataset, valid_dataset = random_split(dataset, [0.9, 0.1])
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True, drop_last=True, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, 4, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)
    for epoch in range(epoch, args.max_epoch + 1):

        adjust_learning_rate(optimizer, epoch, args.lr)

        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size), ncols=150)
        tq.set_description(f'Epoch {epoch}')

        model.train()

        train_main_loss = AverageMeter()
        train_edge_loss = AverageMeter()
        train_seg_loss = AverageMeter()
        train_att_loss = AverageMeter()
        train_dual_loss = AverageMeter()

        for i, (input, target, edge) in enumerate(train_loader):
            batch_pixel_size = input.size(0) * input.size(2) * input.size(3)

            input_var  = Variable(input).cuda()
            target_var = Variable(target).cuda()
            edge_var = Variable(edge).cuda()

            optimizer.zero_grad()

            main_loss = None
            loss_dict = None

            if args.joint_edgeseg_loss:
                # print(input_var.shape)
                # print(target_var.shape)
                # print(edge_var.shape)
                loss_dict = model(input_var, gts=(target_var, edge_var))
                
                if args.seg_weight > 0:
                    log_seg_loss = loss_dict['seg_loss'].mean().clone().detach_()
                    train_seg_loss.update(log_seg_loss.item(), batch_pixel_size)
                    main_loss = loss_dict['seg_loss']

                if args.edge_weight > 0:
                    log_edge_loss = loss_dict['edge_loss'].mean().clone().detach_()
                    train_edge_loss.update(log_edge_loss.item(), batch_pixel_size)
                    if main_loss is not None:
                        main_loss += loss_dict['edge_loss']
                    else:
                        main_loss = loss_dict['edge_loss']
                
                if args.att_weight > 0:
                    log_att_loss = loss_dict['att_loss'].mean().clone().detach_()
                    train_att_loss.update(log_att_loss.item(), batch_pixel_size)
                    if main_loss is not None:
                        main_loss += loss_dict['att_loss']
                    else:
                        main_loss = loss_dict['att_loss']

                if args.dual_weight > 0:
                    log_dual_loss = loss_dict['dual_loss'].mean().clone().detach_()
                    train_dual_loss.update(log_dual_loss.item(), batch_pixel_size)
                    if main_loss is not None:
                        main_loss += loss_dict['dual_loss']
                    else:
                        main_loss = loss_dict['dual_loss']

            else:
                main_loss = model(input_var, gts=target_var)

            main_loss = main_loss.mean()
            log_main_loss = main_loss.clone().detach_()

            train_main_loss.update(log_main_loss.item(), batch_pixel_size)
            tq.set_postfix(main_loss='{:.5f}'.format(train_main_loss.avg),
                           seg_loss='{:.5f}'.format(train_seg_loss.avg),
                            edge_acc='{:.5f}'.format(train_edge_loss.avg))
            tq.update(args.batch_size)

            main_loss.backward()

            optimizer.step()

        # total_iter += len(train_loader)
        logger.log_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
        logger.log_scalar('train/main_loss', train_main_loss.avg, epoch)
        logger.log_scalar('train/seg_loss', train_seg_loss.avg, epoch)
        logger.log_scalar('train/edge_loss', train_edge_loss.avg, epoch)
        valid_metrics = validation(model, valid_loader, criterion_val)
        valid_loss = valid_metrics['valid_loss']
        logger.log_scalar('valid/loss', valid_loss, epoch)
        print(f'\tvalid_loss = {valid_loss:.5f}')
        tq.close()

        #save the model of the current epoch
        epoch_model_path = os.path.join(*[args.model_dir, f'gscnn_epoch_{epoch}.pt'])
        if(epoch % 5 == 0):
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'valid_loss': valid_loss,
                'train_loss': train_main_loss.avg
            }, epoch_model_path)

        if valid_loss < min_val_los:
            min_val_los = valid_loss

            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'valid_loss': valid_loss,
                'train_loss': train_main_loss.avg
            }, best_model_path)

def validate(model, val_loader, criterion):
    tq = tqdm.tqdm(total=(len(val_loader) * 4), ncols=100)
    tq.set_description(f'validating:')
    
    val_loss = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (input, target, edge) in enumerate(val_loader):
            batch_pixel_size = input.size(0) * input.size(2) * input.size(3)

            input_var = Variable(input).cuda()
            target_var = Variable(target).cuda()
            edge_var = Variable(edge).cuda()

            seg_out, edge_out = model(input_var)
            if args.joint_edgeseg_loss:
                loss_dict = criterion((seg_out, edge_out), (target_var, edge_var))
                # print(loss_dict)
                val_loss.update(sum(loss_dict.values()).item(), batch_pixel_size)
            else:
                val_loss.update(criterion(seg_out, target_var).item(), batch_pixel_size)

            tq.set_postfix(val_loss='{:.5f}'.format(val_loss.avg))
            tq.update(4)

    return {'valid_loss': val_loss.avg}

def predict(test_loader, model, latest_model_path, save_dir = './result/test_loader', device = torch.device("cuda:0"), vis_sample_id = None):
    if save_dir != '':
        os.makedirs(save_dir, exist_ok=True)

    model.eval()
    metrics=[]
    pred_list = []
    gt_list = []
    denorm = Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    bar = tqdm.tqdm(total=len(test_loader), ncols=100)
    with torch.no_grad():
        for idx, (img, lab, edge) in enumerate(test_loader, 1):
            # val_data  = Variable(img).cuda()
            val_data = Variable(img).to(device)
            pred, edge_out= model(val_data)

            image = (denorm(img) * 255).squeeze(0).contiguous().cpu().numpy()
            image = image.transpose(2, 1, 0).astype(np.uint8)
            n, c, h, w = pred.size()
            temp_inputs = torch.softmax(pred.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
            preds = torch.gt(temp_inputs[...,1], temp_inputs[...,0]).int().squeeze(-1)
            mask = preds.squeeze(0).view(h,w).contiguous().cpu().numpy()
            mask = np.expand_dims(mask, axis=0)
            # mask = torch.sigmoid(pred.squeeze(0)).contiguous().cpu().numpy()
            label = lab.contiguous().cpu().numpy()
            
            mask = (mask.transpose(2, 1, 0)*255).astype('uint8')
            label = (label.transpose(2, 1, 0)*255).astype('uint8')

            mask[mask>127] = 255
            # label[label>0] = 255
            
            pred_list.append(mask)
            gt_list.append(label)

            if idx in vis_sample_id:

                zeros = np.zeros(mask.shape)
                mask = np.concatenate((mask,zeros,zeros),axis=-1).astype(np.uint8)
                label = np.concatenate((zeros,zeros,label),axis=-1).astype(np.uint8)
                
                temp = cv2.addWeighted(label,1,mask,1,0)
                res = cv2.addWeighted(image,0.6,temp,0.4,0)

                # n, c, h, w = edge_out.size()
                edge_pred = torch.sigmoid(edge_out.squeeze(0)).contiguous().cpu().numpy().transpose(2, 1, 0)
                edge_mask = np.where(edge_pred > 0.7, edge_pred, zeros)
                # edge_mask[edge_mask > 127] = 255
                edge_mask = np.concatenate((zeros,zeros,edge_mask * 255),axis=-1).astype(np.uint8)
                res_edge = cv2.addWeighted(image,0.6,edge_mask,0.4,0)

                cv2.imwrite(os.path.join(save_dir,'%d_test_seg.png' % idx), res)
                cv2.imwrite(os.path.join(save_dir,'%d_test_edge.png' % idx), res_edge)

            bar.update(1)
    bar.close

    # for i in range(1, 10):
                # threshold = i / 10
    metric = calc_metric(pred_list, gt_list, mode='list', threshold=0.5, max_value=255)
    print(metric)
    # if len(metrics) < i:
    #     metrics.append(metric)
    # else:
    #     metrics[i-1]['accuracy'] += metric['accuracy']
    #     metrics[i-1]['precision'] += metric['precision']
    #     metrics[i-1]['recall'] += metric['recall']
    #     metrics[i-1]['f1'] += metric['f1']
    # print(metrics)
    d = datetime.datetime.today()
    datetime.datetime.strftime(d,'%Y-%m-%d %H-%M-%S')
    os.makedirs('./result_dir', exist_ok=True)
    with open(os.path.join('./result_dir', str(d)+'.txt'), 'a', encoding='utf-8') as fout:
                fout.write(str(latest_model_path)+'\n')
                # for i in range(1, 10): 
                line =  "threshold:0.5 | accuracy:{:.5f} | precision:{:.5f} | recall:{:.5f} | f1:{:.5f} " \
                    .format(metric['accuracy'],  metric['precision'],  metric['recall'],  metric['f1']) + '\n'
                fout.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--max_epoch', default=50, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', default=0.005, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--print_freq', default=100, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--weight_decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--batch_size',  default=32, type=int,  help='weight decay (default: 1e-4)')
    parser.add_argument('--num_workers', default=2, type=int, help='output dataset directory')
    parser.add_argument('--num_classes', default=1, type=int, help='output category')
    parser.add_argument('--joint_edgeseg_loss', action='store_true', default=True,help='joint loss')
    parser.add_argument('--img_wt_loss', action='store_true', default=False,help='per-image class-weighted loss')
    parser.add_argument('--edge_weight', type=float, default=1.0,help='Edge loss weight for joint loss')
    parser.add_argument('--seg_weight', type=float, default=1.0,help='Segmentation loss weight for joint loss')
    parser.add_argument('--att_weight', type=float, default=1.0,help='Attention loss weight for joint loss')
    parser.add_argument('--dual_weight', type=float, default=1.0,help='Dual loss weight for joint loss')
    # parser.add_argument('--data_dir',type=str, help='input dataset directory')
    # /home/wj/dataset/seg_dataset /nfs/wj/DamCrack /nfs/wj/192_255_segmentation
    parser.add_argument('--model_dir', type=str, default='/home/wj/local/crack_segmentation/GSCNN/checkpoints/cutDataset', help='output dataset directory')
    parser.add_argument('--arch', type=str, default='network.gscnn.GSCNN')
    parser.add_argument('--trunk', type=str, default='resnet50', help='trunk model, can be: resnet101 (default), resnet50')
    parser.add_argument('-wb', '--wt_bound', type=float, default=1.0)
    parser.add_argument('--syncbn', action='store_true', default=False, help='Synchronized BN')

    parser.add_argument('--sgd', action='store_true', default=True)
    parser.add_argument('--sgd_finetuned',action='store_true',default=False)
    parser.add_argument('--adam', action='store_true', default=False)
    parser.add_argument('--amsgrad', action='store_true', default=False)
    parser.add_argument('--lr_schedule', type=str, default='poly',help='name of lr schedule: poly')
    parser.add_argument('--poly_exp', type=float, default=1.0,help='polynomial LR exponent')
    # parser.add_argument('--snapshot', type=str, default=None)
    parser.add_argument('--snapshot', type=str, default='/home/wj/local/crack_segmentation/GSCNN/checkpoints/pretrain/gscnn_initial_epoch_50.pt')
    parser.add_argument('--restore_optimizer', action='store_true', default=False)
    args = parser.parse_args()
    args.dataset_cls = crack
    os.makedirs(args.model_dir, exist_ok=True)

    # 第一阶段训练
    TRAIN_IMG  = os.path.join(cfg.DATASET.CITYSCAPES_DIR, 'imgs')
    TRAIN_MASK = os.path.join(cfg.DATASET.CITYSCAPES_DIR, 'masks')
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

    dataset = crack.CrackDataSet(img_dir=TRAIN_IMG, img_fnames=train_img_names, img_transform=train_tfms, mask_dir=TRAIN_MASK, mask_fnames=train_mask_names, mask_transform=mask_tfms)
    _dataset, test_dataset = random_split(dataset, [0.9, 0.1],torch.Generator().manual_seed(42))
    test_loader = torch.utils.data.DataLoader(test_dataset, 1, shuffle=False, pin_memory=False)


    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device = torch.device("cuda")
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    num_gpu = torch.cuda.device_count()

    criterion, criterion_val = loss.get_loss(args)
    model = network.get_net(args, criterion)
    optim, scheduler = optimizer.get_optimizer(args, model)

    long_id = 'damcrack_%s_%s' % (str(args.lr), datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    logger = BoardLogger(long_id)
    train(_dataset, model, criterion, optim, validate, args, logger, criterion_val)

    np.random.seed(0)
    vis_sample_id = np.random.randint(0, len(test_loader), 100, np.int32)  # sample idxs for visualization

    # latest_model_path = find_latest_model_path(args.model_dir)
    latest_model_path = os.path.join(*[args.model_dir, 'model_best.pt'])
    state = torch.load(latest_model_path)
    epoch = state['epoch']
    model.load_state_dict(state['state_dict'])
    # weights = state['model']
    # weights_dict = {}
    # for k, v in weights.items():
    #     new_k = k.replace('module.', '') if 'module' in k else k
    #     weights_dict[new_k] = v
    # model.load_state_dict(weights_dict)

    predict(test_loader, model, latest_model_path, os.path.join(args.model_dir, 'test_visual'), device=device, vis_sample_id=vis_sample_id)

