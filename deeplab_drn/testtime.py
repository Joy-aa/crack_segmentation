from nets.unet import Unet
from nets.DSCNet import DSCNet
import torch
import time
from modeling.deeplab import DeepLab
import os
import thop

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0])) 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
print(torch.cuda.device_count())

model = Unet()
# model = DeepLab(backbone='drn', output_stride=16)
# model.load_state_dict(torch.load("/home/wj/local/crack_segmentation/deeplab_drn/deeplabv3+_best_value.pth", map_location=device))
# model = DSCNet(3, 2, 15, 1.0, True, device, 16, 1)
# model.load_state_dict(torch.load("/home/wj/local/crack_segmentation/deeplab_drn/DSCNet_best_value.pth"))
model.eval()
model = model.to(device)
dump_input = torch.ones(1,3,1024,1024).to(device)

output = model(dump_input)
print(output.size())

# flops, params = thop.profile(model, inputs=(dump_input, )) 

# flops, params = thop.clever_format([flops, params], "%.3f") 
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