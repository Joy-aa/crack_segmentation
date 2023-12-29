from thop import profile
from thop import clever_format
import torch
import sys
sys.path.append("/home/wj/local/crack_segmentation")
from unet.network.unet_gate import UNetGate
from unet.network.unet_transfer import UNet16
import os
import time



# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0])) 
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
print(torch.cuda.device_count())

# model = UNetGate().cuda()
model = UNet16()
model.to(device)
dump_input = torch.ones(1,3,224,224).to(device)

# Warn-up
for _ in range(10):
    start = time.time()
    outputs = model(dump_input)
    torch.cuda.synchronize()
    end = time.time()
    print('Time:{}ms'.format((end-start)*1000))

with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=False, profile_memory=False) as prof:
    outputs = model(dump_input)
print(prof.table())