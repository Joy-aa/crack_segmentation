
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
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import argparse
from nets.SDDNet import SDDNet
from nets.STRNet import STRNet
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id


def main(model, device):
    # ----------------------- dataset ----------------------- #

    DIR_IMG  = os.path.join(cfg.data_dir, 'images')
    DIR_MASK = os.path.join(cfg.data_dir, 'masks')

    img_names  = [path.name for path in Path(DIR_IMG).glob('*.jpg')]
    mask_names = [path.name for path in Path(DIR_MASK).glob('*.jpg')]

    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(channel_means, channel_stds)])

    mask_tfms = transforms.Compose([transforms.ToTensor()])


    dataset = ImgDataSet(img_dir=DIR_IMG, img_fnames=img_names, img_transform=train_tfms, mask_dir=DIR_MASK, mask_fnames=mask_names, mask_transform=mask_tfms)
    train_size = int(0.85*len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=4, drop_last=True)

    val_loader = DataLoader(valid_dataset, batch_size=cfg.val_batch_size, shuffle=False, num_workers=4, drop_last=True)

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
            model.train()

            # ---------------------  training ------------------- #
            bar = tqdm(enumerate(train_loader), total=(len(train_loader) * cfg.train_batch_size))
            bar.set_description('Epoch %d --- Training --- :' % epoch)
            train_loss = {
                        'total_loss': 0,
                        'output_loss': 0,
                        'eval_fuse5_loss': 0,
                        'eval_fuse4_loss': 0,
                        'eval_fuse3_loss': 0,
                        'eval_fuse2_loss': 0,
                        'eval_fuse1_loss': 0,
            }
            for idx, (img, lab) in bar:
                # data, target = img.type(torch.cuda.FloatTensor).to(device), lab.type(torch.cuda.FloatTensor).to(device)
                data  = Variable(img).cuda()
                target = Variable(lab).cuda()
                pred = trainer.train_op(data, target)

                train_loss['total_loss'] += trainer.log_loss['total_loss']
                train_loss['output_loss'] += trainer.log_loss['output_loss']
                bar.set_postfix(total_loss='{:.5f}'.format(train_loss['total_loss']), output_loss='{:.5f}'.format(train_loss['output_loss']))
                bar.update(cfg.train_batch_size)
                
            bar.close()
            bar1 = tqdm(enumerate(val_loader, 1), total=(len(val_loader) * cfg.val_batch_size))
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
                        for idx, (img, lab) in bar1:
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

    model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    model.to(device)
    main(model, device)
