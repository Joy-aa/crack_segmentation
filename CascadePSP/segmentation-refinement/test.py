import cv2
import time
import matplotlib.pyplot as plt
import segmentation_refinement as refine


image = cv2.imread('/mnt/hangzhou_116_homes/DamDetection/data/image/creak_s120.JPG')
mask = cv2.imread('/home/wj/local/crack_segmentation/unet/result3/creak_s120.jpg', cv2.IMREAD_GRAYSCALE)
# print(image.shape)
# print(mask.shape)
# model_path can also be specified here
# This step takes some time to load the model
refiner = refine.Refiner(device='cuda:0', model_folder='/home/wj/local/crack_segmentation/CascadePSP/checkpoints') # device can also be 'cpu'

# Fast - Global step only.
# Smaller L -> Less memory usage; faster in fast mode.
output = refiner.refine(image, mask, fast=False, L=900) 

# this line to save output
cv2.imwrite('/home/wj/local/crack_segmentation/CascadePSP/segmentation-refinement/test/output.png', output)

# plt.imshow(output)
# plt.show()
