from thop import profile
from thop import clever_format
import torch
import sys
sys.path.append("/home/wj/local/crack_segmentation")
from unet.network.unet_gate import UNetGate
from unet.network.unet_transfer import UNet16
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = torch.device("cuda")
input = torch.randn(1, 3, 224, 224).cuda()

# model = UNetGate().cuda()
model = UNet16().cuda()
# model = Model() 

flops, params = profile(model, inputs=(input, )) 

flops, params = clever_format([flops, params], "%.3f") 
print(flops)
print(params)
