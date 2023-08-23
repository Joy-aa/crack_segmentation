from  cityscape  import DatasetTrain ,DatasetVal
import argparse
from torch.utils.data import  DataLoader
from pathlib import Path
import yaml
from train import Trainer
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.optim import lr_scheduler
from model.dfanet import DFANet,load_backbone
from config import Config
from loss import CrossEntropyLoss2d
import os
import torch
import sys
sys.path.append("/home/wj/local/crack_segmentation")
from data_loader import ImgDataSet
from LossFunctions import dice_loss
import torchvision.transforms as transforms


if __name__=='__main__':

    cfg=Config()
    #create dataset
    # train_dataset = DatasetTrain(cityscapes_data_path="/raid/DataSet/Cityscape",
    #                             cityscapes_meta_path="/raid/DataSet/Cityscape/gtFine/")
    # val_dataset = DatasetVal(cityscapes_data_path="/raid/DataSet/Cityscape",
    #                          cityscapes_meta_path="/raid/DataSet/Cityscape/gtFine")       
    # train_loader = DataLoader(dataset=train_dataset,
    #                                        batch_size=16, shuffle=True,
    #                                        num_workers=8)
    # val_loader = DataLoader(dataset=val_dataset,
    #                                      batch_size=16, shuffle=False, 
    #                                      num_workers=8)

    data_dir='/home/wj/dataset/seg_dataset/'
    
    TRAIN_IMG  = os.path.join(data_dir, 'images')
    TRAIN_MASK = os.path.join(data_dir, 'masks')
    # TRAIN_IMG  = os.path.join(data_dir, 'train_image')
    # TRAIN_MASK = os.path.join(data_dir, 'train_label')
    # VALID_IMG = os.path.join(data_dir, 'val_image')
    # VALID_MASK = os.path.join(data_dir, 'val_label')


    train_img_names  = [path.name for path in Path(TRAIN_IMG).glob('*.jpg')]
    train_mask_names = [path.name for path in Path(TRAIN_MASK).glob('*.jpg')]
    # valid_img_names  = [path.name for path in Path(VALID_IMG).glob('*.jpg')]
    # valid_mask_names = [path.name for path in Path(VALID_MASK).glob('*.bmp')]


    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(channel_means, channel_stds)])

    val_tfms = transforms.Compose([transforms.ToTensor()])

    mask_tfms = transforms.Compose([transforms.ToTensor()])

    train_dataset = ImgDataSet(img_dir=TRAIN_IMG, img_fnames=train_img_names, img_transform=val_tfms, mask_dir=TRAIN_MASK, mask_fnames=train_mask_names, mask_transform=mask_tfms)
    # valid_dataset = ImgDataSet(img_dir=VALID_IMG, img_fnames=valid_img_names, img_transform=val_tfms, mask_dir=VALID_MASK, mask_fnames=valid_mask_names, mask_transform=mask_tfms)
    train_size = int(0.4*len(train_dataset))
    rest_size = len(train_dataset) - train_size
    train_dataset, rest_dataset = torch.utils.data.random_split(train_dataset, [train_size, rest_size])

    valid_size = int(0.2*len(train_dataset))
    train_size = len(train_dataset) - valid_size
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

    # train_loader = DataLoader(train_dataset, 4, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=4)
    # valid_loader = DataLoader(valid_dataset, 4, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=4)
    train_loader = DataLoader(train_dataset, 4, shuffle=True, pin_memory=False, num_workers=4)
    valid_loader = DataLoader(valid_dataset, 4, shuffle=True, pin_memory=False, num_workers=4)

    print(f'total train loader = {len(train_loader)}')                               
    
    net = DFANet(cfg.ENCODER_CHANNEL_CFG,decoder_channel=64,num_classes=1)
    # device = torch.device("cuda")
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # num_gpu = torch.cuda.device_count()

    # net = torch.nn.DataParallel(net, device_ids=range(num_gpu))
    # net.to(device)
    # net = load_backbone(net,"backbone.pth")

    #load loss
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9,weight_decay=0.00001)  #select the optimizer

    lr_fc=lambda iteration: (1-iteration/400000)**0.9

    exp_lr_scheduler = lr_scheduler.LambdaLR(optimizer,lr_fc,-1)
    
    trainer = Trainer('training', optimizer,exp_lr_scheduler, net, cfg, './log')
    trainer.load_weights(trainer.find_last())
    trainer.train(train_loader, valid_loader, criterion, 1500)
    # trainer.evaluate(valid_loader)
    print('Finished Training')
