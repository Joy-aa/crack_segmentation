from thop import profile
from thop import clever_format
import torch
import network
import loss
import argparse
from datasets import crack
import os
import time
import torch
import inspect

from torchvision import models
# from gpu_mem_track import MemTracker  # 引用显存跟踪代码

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--max_epoch', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', default=0.0001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--print_freq', default=100, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--weight_decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--batch_size',  default=16, type=int,  help='weight decay (default: 1e-4)')
parser.add_argument('--num_workers', default=2, type=int, help='output dataset directory')
parser.add_argument('--num_classes', default=1, type=int, help='output category')
parser.add_argument('--joint_edgeseg_loss', action='store_true', default=True,help='joint loss')
parser.add_argument('--img_wt_loss', action='store_true', default=False,help='per-image class-weighted loss')
parser.add_argument('--edge_weight', type=float, default=1.0,help='Edge loss weight for joint loss')
parser.add_argument('--seg_weight', type=float, default=1.0,help='Segmentation loss weight for joint loss')
parser.add_argument('--att_weight', type=float, default=0,help='Attention loss weight for joint loss')
parser.add_argument('--dual_weight', type=float, default=0,help='Dual loss weight for joint loss')
# parser.add_argument('--data_dir',type=str, help='input dataset directory')
# /home/wj/dataset/seg_dataset /nfs/wj/DamCrack /nfs/wj/192_255_segmentation
parser.add_argument('--model_dir', type=str, default='/home/wj/local/crack_segmentation/GSCNN/checkpoints/dice_loss', help='output dataset directory')
parser.add_argument('--arch', type=str, default='network.gscnn.GSCNN')
parser.add_argument('--trunk', type=str, default='resnet50', help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0)

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

# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0])) 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
print(torch.cuda.device_count())

dump_input = torch.ones(1,3,1024,1024).to(device)

criterion = torch.nn.BCEWithLogitsLoss()
model = network.get_model(network='network.gscnn.GSCNN', num_classes=2, criterion=criterion, trunk='resnet50')
model.eval()
model.to(device)

output = model(dump_input)
print(output.size())

# flops, params = profile(model, inputs=(dump_input, )) 
# flops, params = clever_format([flops, params], "%.3f") 
# print(flops)
# print(params)


# Warn-up
total_mean = 0
for _ in range(10):
    start = time.time()
    outputs = model(dump_input)
    torch.cuda.synchronize()
    end = time.time()
    if _ > 0:
        total_mean += (end-start)*1000
    print('Time:{}ms'.format((end-start)*1000))

with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=False, profile_memory=False) as prof:
    outputs = model(dump_input)
print(prof.table())
print(total_mean/9)