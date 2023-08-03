
from tqdm import tqdm
from nets.crackformerII import crackformer
from utils.trainer import Trainer
from config import Config as cfg
import numpy as np
import torch
import os
import cv2
import sys
from pathlib import Path
sys.path.append('/home/wj/local/crack_segmentation')
from data_loader import ImgDataSet
from logger import BoardLogger
import datetime
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import argparse
from nets.SDDNet import SDDNet
from nets.STRNet import STRNet
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id


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
    if epoch == 20 or epoch == 50 or epoch == 100:
        lr = lr * cfg.lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def main(model, device):
    # ----------------------- dataset ----------------------- #

    # TRAIN_IMG  = os.path.join(cfg.data_dir, 'train_image')
    # TRAIN_MASK = os.path.join(cfg.data_dir, 'train_label')
    # VALID_IMG = os.path.join(cfg.data_dir, 'val_image')
    # VALID_MASK = os.path.join(cfg.data_dir, 'val_label')


    # train_img_names  = [path.name for path in Path(TRAIN_IMG).glob('*.jpg')]
    # train_mask_names = [path.name for path in Path(TRAIN_MASK).glob('*.bmp')]
    # valid_img_names  = [path.name for path in Path(VALID_IMG).glob('*.jpg')]
    # valid_mask_names = [path.name for path in Path(VALID_MASK).glob('*.bmp')]
    # # train_img_names = train_img_names[:len(train_img_names)*0.1]
    # # train_mask_names = train_mask_names[:len(train_mask_names)*0.1]
    # print(f'total train images = {len(train_img_names)}')
    # print(f'total valid images = {len(valid_img_names)}')

    TRAIN_IMG  = os.path.join(cfg.data_dir, 'imgs')
    TRAIN_MASK = os.path.join(cfg.data_dir, 'masks')


    train_img_names  = [path.name for path in Path(TRAIN_IMG).glob('*.png')]
    train_mask_names = [path.name for path in Path(TRAIN_MASK).glob('*.png')]
    # train_img_names = train_img_names[:len(train_img_names)*0.1]
    # train_mask_names = train_mask_names[:len(train_mask_names)*0.1]
    print(f'total train images = {len(train_img_names)}')
    
    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose([transforms.ToTensor()])
                                    #  transforms.Normalize(channel_means, channel_stds)])

    val_tfms = transforms.Compose([transforms.ToTensor()])
                                #    transforms.Normalize(channel_means, channel_stds)])
    
    mask_tfms = transforms.Compose([transforms.ToTensor()])


    train_dataset = ImgDataSet(img_dir=TRAIN_IMG, img_fnames=train_img_names, img_transform=train_tfms, mask_dir=TRAIN_MASK, mask_fnames=train_mask_names, mask_transform=mask_tfms)
    # valid_dataset = ImgDataSet(img_dir=VALID_IMG, img_fnames=valid_img_names, img_transform=val_tfms, mask_dir=VALID_MASK, mask_fnames=valid_mask_names, mask_transform=mask_tfms)
    # train_size = int(0.2*len(train_dataset))
    # rest_size = len(train_dataset) - train_size
    # train_dataset, rest_dataset = torch.utils.data.random_split(train_dataset, [train_size, rest_size])

    _dataset, test_dataset = random_split(train_dataset, [0.9, 0.1],torch.Generator().manual_seed(42))
    # _dataset, tmp_dataset = random_split(_dataset, [0.1, 0.9],torch.Generator().manual_seed(42))
    train_dataset, valid_dataset = random_split(_dataset, [0.9, 0.1],torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(train_dataset, cfg.train_batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=4)
    val_loader = torch.utils.data.DataLoader(valid_dataset, cfg.val_batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, cfg.test_batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=4)

    # -------------------- build trainer --------------------- #
    long_id = '%s_%s' % (str(cfg.lr), datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    logger = BoardLogger(long_id)

    # -------------------- build trainer --------------------- #

    trainer = Trainer(model).to(device)

    if cfg.pretrained_model:
        pretrained_dict = trainer.saver.load(cfg.pretrained_model, multi_gpu=True)
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    try:

        for epoch in range(1, cfg.epoch):

            adjust_learning_rate(trainer.optimizer, epoch, cfg.lr)   

            trainer.vis.log('Start Epoch %d ...' % epoch, 'train info')
            model.train()

            # ---------------------  training ------------------- #
            bar = tqdm(total=(len(train_loader) * cfg.train_batch_size))
            bar.set_description('Epoch %d --- Training --- :' % epoch)
            train_total_loss = AverageMeter()
            train_output_loss = AverageMeter()
            for idx, (img, lab) in enumerate(train_loader, 1):
                # data, target = img.type(torch.cuda.FloatTensor).to(device), lab.type(torch.cuda.FloatTensor).to(device)
                data  = Variable(img).cuda()
                target = Variable(lab).cuda()
                pred = trainer.train_op(data, target)

                if idx % cfg.vis_train_loss_every == 0:
                    trainer.vis.log(trainer.log_loss, 'train_loss')
                    trainer.vis.plot_many({
                        'train_total_loss': trainer.log_loss['total_loss'],
                        'train_output_loss': trainer.log_loss['output_loss'],
                        'train_fuse5_loss': trainer.log_loss['fuse5_loss'],
                        'train_fuse4_loss': trainer.log_loss['fuse4_loss'],
                        'train_fuse3_loss': trainer.log_loss['fuse3_loss'],
                        'train_fuse2_loss': trainer.log_loss['fuse2_loss'],
                        'train_fuse1_loss': trainer.log_loss['fuse1_loss'],
                    })

                if idx % cfg.vis_train_acc_every == 0:
                    trainer.acc_op(pred[0], target)
                    trainer.vis.log(trainer.log_acc, 'train_acc')
                    trainer.vis.plot_many({
                        'train_mask_acc': trainer.log_acc['mask_acc'],
                        'train_mask_pos_acc': trainer.log_acc['mask_pos_acc'],
                        'train_mask_neg_acc': trainer.log_acc['mask_neg_acc'],
                    })
                if idx % cfg.vis_train_img_every == 0:
                    trainer.vis.img_many({
                        'train_img': data.cpu(),
                        'train_output': torch.sigmoid(pred[0].contiguous().cpu()),
                        'train_lab': target.cpu(),
                        'train_fuse5': torch.sigmoid(pred[1].contiguous().cpu()),
                        'train_fuse4': torch.sigmoid(pred[2].contiguous().cpu()),
                        'train_fuse3': torch.sigmoid(pred[3].contiguous().cpu()),
                        'train_fuse2': torch.sigmoid(pred[4].contiguous().cpu()),
                        'train_fuse1': torch.sigmoid(pred[5].contiguous().cpu()),
                    })

                 # train_loss['total_loss'] += trainer.log_loss['total_loss']
                train_total_loss.update(trainer.log_loss['total_loss'])
                # train_loss['output_loss'] += trainer.log_loss['output_loss']
                train_output_loss.update(trainer.log_loss['output_loss'])
                bar.set_postfix(total_loss='{:.5f}'.format(train_total_loss.avg), output_loss='{:.5f}'.format(train_output_loss.avg))
                bar.update(cfg.train_batch_size)
                
            bar.close()
            bar1 = tqdm(total=(len(val_loader) * cfg.val_batch_size))
            bar1.set_description('Epoch %d --- Evaluation --- :' % epoch)
            # -------------------- val ------------------- #
            model.eval()
            val_loss = {
                        'eval_total_loss': 0,
                        'eval_output_loss': 0,
                        'eval_fuse5_loss': 0,
                        'eval_fuse4_loss': 0,
                        'eval_fuse3_loss': 0,
                        'eval_fuse2_loss': 0,
                        'eval_fuse1_loss': 0,}
            val_acc = {
                        'mask_acc': 0,
                        'mask_pos_acc': 0,
                        'mask_neg_acc': 0,}

            with torch.no_grad():
                        for idx, (img, lab) in enumerate(val_loader, 1):
                            # val_data, val_target = img.type(torch.cuda.FloatTensor).to(device), lab.type(torch.cuda.FloatTensor).to(device)
                            val_data  = Variable(img).cuda()
                            val_target = Variable(lab).cuda()
                            val_pred = trainer.val_op(val_data, val_target)
                            trainer.acc_op(val_pred[0], val_target)
                            val_loss['eval_total_loss'] += trainer.log_loss['total_loss']
                            val_loss['eval_output_loss'] += trainer.log_loss['output_loss']
                            val_loss['eval_fuse5_loss'] += trainer.log_loss['fuse5_loss']
                            val_loss['eval_fuse4_loss'] += trainer.log_loss['fuse4_loss']
                            val_loss['eval_fuse3_loss'] += trainer.log_loss['fuse3_loss']
                            val_loss['eval_fuse2_loss'] += trainer.log_loss['fuse2_loss']
                            val_loss['eval_fuse1_loss'] += trainer.log_loss['fuse1_loss']
                            val_acc['mask_acc'] += trainer.log_acc['mask_acc']
                            val_acc['mask_pos_acc'] += trainer.log_acc['mask_pos_acc']
                            val_acc['mask_neg_acc'] += trainer.log_acc['mask_neg_acc']

                            bar1.set_postfix(total_loss='{:.5f}'.format(val_loss['eval_total_loss']),
                                            output_loss='{:.5f}'.format(val_loss['eval_output_loss']),
                                            total_acc='{:.5f}'.format(val_acc['mask_acc'] / idx),
                                            pos_acc='{:.5f}'.format(val_acc['mask_pos_acc'] / idx))
                            bar1.update(cfg.val_batch_size)
                        else:
                            # ----------------- save model ---------------- #
                            if cfg.save_pos_acc < (val_acc['mask_pos_acc'] / idx):
                                cfg.save_pos_acc = (val_acc['mask_pos_acc'] / idx)
                                cfg.save_acc = (val_acc['mask_acc'] / idx)
                                trainer.saver.save(model, tag='epoch(%d)_acc(%0.2f-%0.2f)' % (epoch, cfg.save_pos_acc, cfg.save_acc))
            bar1.close()
            if epoch != 0:
                trainer.saver.save(model, tag='%s_epoch(%d)' % (cfg.name, epoch))
            logger.log_scalar('train/lr', trainer.optimizer.param_groups[0]['lr'], epoch)
            logger.log_scalar('train/total_loss', train_total_loss.avg, epoch)
            logger.log_scalar('train/output_loss', train_output_loss.avg, epoch)
            logger.log_scalar('valid/total_loss', val_loss['eval_total_loss'], epoch)
            logger.log_scalar('valid/output_loss', val_loss['eval_output_loss'], epoch)

    except KeyboardInterrupt:

        trainer.saver.save(model, tag='Auto_Save_Model')
        print('\n Catch KeyboardInterrupt, Auto Save final model : %s' % trainer.saver.show_save_pth_name)
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == '__main__':

    if cfg.name == 'crackformer':
        model = crackformer()
    elif cfg.name  == 'SDDNet':
        model = SDDNet(3, 1)
    elif cfg.name  == 'STRNet':
        model = STRNet(3, 1)
    else:
        print('undefind model name pattern')
        exit()
    # model.load_state_dict(torch.load(args.model_path))
    
    device = torch.device("cuda")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_gpu = torch.cuda.device_count()
    print(device)

    model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    model.to(device)
    main(model, device)
