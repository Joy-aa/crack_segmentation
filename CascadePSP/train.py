import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset

from models.psp.pspnet import PSPNet
from models.sobel_op import SobelComputer
from dataset import OnlineTransformDataset, OfflineDataset
from util.logger import BoardLogger
from util.model_saver import ModelSaver
from util.hyper_para import HyperParameters
from util.log_integrator import Integrator
from util.metrics_compute import compute_loss_and_metrics, iou_hooks_to_be_used
from util.image_saver import vis_prediction
from util.Visdom import Visualizer

import time
import os
import datetime

torch.backends.cudnn.benchmark = True

# Parse command line arguments
para = HyperParameters()
para.parse()

# Logging
if para['id'].lower() != 'null':
    long_id = '%s_%s' % (para['id'],datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
else:
    long_id = None
logger = BoardLogger(long_id)
logger.log_string('hyperpara', str(para))

# vis = Visualizer(env='cascadPpsp')

print('CUDA Device count: ', torch.cuda.device_count())

# Construct model
model = PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50')
model = nn.DataParallel(
        model.cuda(), device_ids=[0,1]
    )
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# model.to(device)

if para['load'] is not None:
    model.load_state_dict(torch.load(para['load']))
optimizer = optim.Adam(model.parameters(), lr=para['lr'], weight_decay=para['weight_decay'])

# train_dataset = ConcatDataset([fss_dataset, duts_tr_dataset, duts_te_dataset, ecssd_dataset, msra_dataset])

data_dir = '/nfs/wj/192_255_segmentation/'
# data_dir = '/mnt/hangzhou_116_homes/wj/192_255_segmentation/'
DIR_IMG  = os.path.join(data_dir, 'imgs')
DIR_MASK  = os.path.join(data_dir, 'masks')
DIR_SEG = os.path.join(data_dir, "segs")
# dataset = OnlineTransformDataset(DIR_IMG, DIR_MASK, method=1, perturb=True)
dataset = OfflineDataset(DIR_IMG, DIR_MASK, DIR_SEG, need_name=False, resize=False, do_crop=False)
train_dataset = ConcatDataset([dataset])
print('Total training size: ', len(train_dataset))

# For randomness: https://github.com/pytorch/pytorch/issues/5059
def worker_init_fn(worker_id): 
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Dataloaders, multi-process data loading
train_loader = DataLoader(train_dataset, para['batch_size'], shuffle=True, num_workers=8,
                            worker_init_fn=worker_init_fn, drop_last=True, pin_memory=True)

sobel_compute = SobelComputer()

# Learning rate decay scheduling
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, para['steps'], para['gamma'])

torch.set_num_threads(1)
torch.backends.cudnn.benchmark = True

saver = ModelSaver(long_id)
report_interval = 1000
save_im_interval = 200

total_epoch = int(para['iterations']/len(train_loader) + 0.5)
print('Actual training epoch: ', total_epoch)

train_integrator = Integrator(logger)
# train_integrator.add_hook(iou_hooks_to_be_used)
total_iter = 0
last_time = 0
for e in range(total_epoch):
    np.random.seed() # reset seed
    epoch_start_time = time.time()

    # Train loop
    model = model.train()
    # print(train_loader)
    # for tuple in train_loader:
    #     print(len(tuple))
    #     print(tuple[0].shape)
    for im, seg, gt in train_loader:
        im, seg, gt = im.cuda(), seg.cuda(), gt.cuda()

        total_iter += 1
        if total_iter % 10000 == 0:
            saver.save_model(model, total_iter)

        images = model(im, seg)

        images['im'] = im
        images['seg'] = seg
        images['gt'] = gt

        sobel_compute.compute_edges(images)

        loss_and_metrics = compute_loss_and_metrics(images, para)
        train_integrator.add_dict(loss_and_metrics)

        optimizer.zero_grad()
        (loss_and_metrics['total_loss']).backward()
        optimizer.step()

        if total_iter % report_interval == 0:
            logger.log_scalar('train/lr', scheduler.get_lr()[0], total_iter)
            train_integrator.finalize('train', total_iter)
            train_integrator.reset_except_hooks()

        # Need to put step AFTER get_lr() for correct logging, see issue #22107 in PyTorch
        scheduler.step()

        if total_iter % save_im_interval == 0:
            predict_vis = vis_prediction(images)
            logger.log_cv2('train/predict', predict_vis, total_iter)

# Final save!
saver.save_model(model, total_iter)
