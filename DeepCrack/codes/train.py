from data.augmentation import augCompose, RandomBlur, RandomColorJitter
from data.dataset import readIndex, dataReadPip, loadedDataset
from tqdm import tqdm
from model.deepcrack import DeepCrack
from model.deepcrackv2 import DeepCrackV2
from trainer import DeepCrackTrainer
from config import Config as cfg
import numpy as np
import torch
import os
import cv2
import sys
import torchvision.transforms as transforms
from torch.autograd import Variable
from pathlib import Path
import sys
sys.path.append("/home/wj/local/crack_segmentation")
from data_loader import ImgDataSet

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
    lr = lr * (cfg.lr_decay ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    # ----------------------- dataset ----------------------- #

    TRAIN_IMG  = os.path.join(cfg.train_data_path, 'train_image')
    TRAIN_MASK = os.path.join(cfg.train_data_path, 'train_label')
    VALID_IMG = os.path.join(cfg.train_data_path, 'val_image')
    VALID_MASK = os.path.join(cfg.train_data_path, 'val_label')


    train_img_names  = [path.name for path in Path(TRAIN_IMG).glob('*.jpg')]
    train_mask_names = [path.name for path in Path(TRAIN_MASK).glob('*.bmp')]
    valid_img_names  = [path.name for path in Path(VALID_IMG).glob('*.jpg')]
    valid_mask_names = [path.name for path in Path(VALID_MASK).glob('*.bmp')]

    print(f'total train images = {len(train_img_names)}')
    print(f'total valid images = {len(valid_img_names)}')
    
    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(channel_means, channel_stds)])

    val_tfms = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(channel_means, channel_stds)])
    
    mask_tfms = transforms.Compose([transforms.ToTensor()])

    # data_augment_op = augCompose(transforms=[[RandomColorJitter, 0.5], [RandomBlur, 0.2]])

    # train_pipline = dataReadPip(transforms=data_augment_op)

    # test_pipline = dataReadPip(transforms=None)

    # train_dataset = loadedDataset(readIndex(cfg.train_data_path, shuffle=True), preprocess=train_pipline)

    # test_dataset = loadedDataset(readIndex(cfg.val_data_path), preprocess=test_pipline)

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_batch_size,
    #                                            shuffle=True, num_workers=4, drop_last=True)

    # val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.val_batch_size,
    #                                          shuffle=False, num_workers=4, drop_last=True)
    
    train_dataset = ImgDataSet(img_dir=TRAIN_IMG, img_fnames=train_img_names, img_transform=train_tfms, mask_dir=TRAIN_MASK, mask_fnames=train_mask_names, mask_transform=mask_tfms)
    valid_dataset = ImgDataSet(img_dir=VALID_IMG, img_fnames=valid_img_names, img_transform=val_tfms, mask_dir=VALID_MASK, mask_fnames=valid_mask_names, mask_transform=mask_tfms)
    train_size = int(0.5*len(train_dataset))
    rest_size = len(train_dataset) - train_size
    train_dataset, rest_dataset = torch.utils.data.random_split(train_dataset, [train_size, rest_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, cfg.train_batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=4)
    val_loader = torch.utils.data.DataLoader(valid_dataset, cfg.val_batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=4)

    # -------------------- build trainer --------------------- #

    device = torch.device("cuda")
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # num_gpu = torch.cuda.device_count()

    # model = DeepCrack(num_classes=1)
    initial = True
    # model = DeepCrackV2(pretrained_model=initial)
    model = DeepCrack()
    # model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    model.to(device)

    trainer = DeepCrackTrainer(model).to(device)

    if initial == False:
        pretrained_dict = trainer.saver.load(cfg.pretrained_model, multi_gpu=True)
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        trainer.vis.log('load checkpoint: %s' % cfg.pretrained_model, 'train info')
        epoch_str = Path(cfg.pretrained_model).stem.split('_')[0]
        # print(epoch_str)
        epoch = int(epoch_str[-3:-1])
        # print(epoch)
    else:
        epoch = 0

    try:

        for epoch in range(epoch+1, cfg.epoch):

            adjust_learning_rate(trainer.optimizer, epoch, cfg.lr)            

            trainer.vis.log('Start Epoch %d ...' % epoch, 'train info')
            model.train()

            # ---------------------  training ------------------- #
            bar = tqdm(total=(len(train_loader) * cfg.train_batch_size))
            bar.set_description('Epoch %d --- Training --- :' % epoch)
            # train_loss = {
            #             'total_loss': 0,
            #             'output_loss': 0,
            # }
            train_total_loss = AverageMeter()
            train_output_loss = AverageMeter()
            for idx, (img, lab) in enumerate(train_loader):
                # data, target = img.type(torch.cuda.FloatTensor).to(device), lab.type(torch.cuda.FloatTensor).to(device)
                data  = Variable(img).cuda()
                target = Variable(lab).cuda()
                pred = trainer.train_op(data, target)
                # print(pred[0].shape, pred[1].shape)
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
                        # 'train_img': data.cpu(),
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

                            bar1.set_postfix(total_acc='{:.5f}'.format(val_acc['mask_acc'] / idx),
                                            pos_acc='{:.5f}'.format(val_acc['mask_pos_acc'] / idx))
                            bar1.update(cfg.val_batch_size)
                else:
                            trainer.vis.img_many({
                                # 'eval_img': val_data.cpu(),
                                'eval_output': torch.sigmoid(val_pred[0].contiguous().cpu()),
                                'eval_lab': val_target.cpu(),
                                'eval_fuse5': torch.sigmoid(val_pred[1].contiguous().cpu()),
                                'eval_fuse4': torch.sigmoid(val_pred[2].contiguous().cpu()),
                                'eval_fuse3': torch.sigmoid(val_pred[3].contiguous().cpu()),
                                'eval_fuse2': torch.sigmoid(val_pred[4].contiguous().cpu()),
                                'eval_fuse1': torch.sigmoid(val_pred[5].contiguous().cpu()),

                            })
                            trainer.vis.plot_many({
                                'eval_total_loss': val_loss['eval_total_loss'] / idx,
                                'eval_output_loss': val_loss['eval_output_loss'] / idx,
                                'eval_fuse5_loss': val_loss['eval_fuse5_loss'] / idx,
                                'eval_fuse4_loss': val_loss['eval_fuse4_loss'] / idx,
                                'eval_fuse3_loss': val_loss['eval_fuse3_loss'] / idx,
                                'eval_fuse2_loss': val_loss['eval_fuse2_loss'] / idx,
                                'eval_fuse1_loss': val_loss['eval_fuse1_loss'] / idx,

                            })
                            trainer.vis.plot_many({
                                'eval_mask_acc': val_acc['mask_acc'] / idx,
                                'eval_mask_neg_acc': val_acc['mask_neg_acc'] / idx,
                                'eval_mask_pos_acc': val_acc['mask_pos_acc'] / idx,

                            })
                            # ----------------- save model ---------------- #
                            if cfg.save_pos_acc < (val_acc['mask_pos_acc'] / idx):
                                cfg.save_pos_acc = (val_acc['mask_pos_acc'] / idx)
                                cfg.save_acc = (val_acc['mask_acc'] / idx)
                                trainer.saver.save(model, tag='epoch(%d)_acc(%0.5f-%0.5f)' % (epoch, cfg.save_pos_acc, cfg.save_acc))
                                trainer.vis.log('Save Model %s_epoch(%d)_acc(%0.5f-%0.5f)' % (cfg.name, epoch, cfg.save_pos_acc, cfg.save_acc), 'train info')
            if epoch != 0:
                trainer.saver.save(model, tag='%s_epoch(%d)' % (
                    cfg.name, epoch))
                trainer.vis.log('Save Model -%s_epoch(%d)' % (
                    cfg.name, epoch), 'train info')

    except KeyboardInterrupt:

        trainer.saver.save(model, tag='Auto_Save_Model')
        print('\n Catch KeyboardInterrupt, Auto Save final model : %s' % trainer.saver.show_save_pth_name)
        trainer.vis.log('Catch KeyboardInterrupt, Auto Save final model : %s' % trainer.saver.show_save_pth_name,
                        'train info')
        trainer.vis.log('Training End!!')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == '__main__':
    main()
