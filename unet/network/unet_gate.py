from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
import network.mynn as mynn
import numpy as np
import cv2

input_size = (480, 480)

class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor,
                        mode=self.mode, align_corners=self.align_corners)
        return x

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        # self.activation = nn.ReLU(inplace=True)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                # nn.ReLU(inplace=True)
                nn.GELU()
            )
        else:
            self.block = nn.Sequential(
                Interpolate(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)
    
class Fuse(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.nn = ConvRelu(in_channels, middle_channels)
        self.conv = ConvRelu(middle_channels, out_channels)

    def forward(self, edge_features, texture_features):
        outputs = torch.cat([edge_features, texture_features], 1)
        outputs = self.nn(outputs)
        outputs = self.conv(outputs)
        # outputs = F.interpolate(outputs, scale_factor=2, mode='bilinear')
        return outputs

class GatedSpatialConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        """

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(GatedSpatialConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, 'zeros')

        self._gate_conv = nn.Sequential(
            mynn.Norm2d(in_channels+1),
            nn.Conv2d(in_channels+1, in_channels+1, 1),
            nn.ReLU(), 
            nn.Conv2d(in_channels+1, 1, 1),
            mynn.Norm2d(1),
            nn.Sigmoid()
        )

    def forward(self, input_features, gating_features):
        """

        :param input_features:  [NxCxHxW]  featuers comming from the shape branch (canny branch).
        :param gating_features: [Nx1xHxW] features comming from the texture branch (resnet). Only one channel feature map.
        :return:
        """
        alphas = self._gate_conv(torch.cat([input_features, gating_features], dim=1))

        input_features = (input_features * (alphas + 1)) 
        return F.conv2d(input_features, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
  
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

class UNetGate(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG16
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        #print(torchvision.models.vgg16(pretrained=pretrained))

        self.encoder = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features

        self.relu = nn.GELU()

        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   self.relu,
                                   self.encoder[26],
                                   self.relu,
                                   self.encoder[28],
                                   self.relu)

        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.canny_conv = nn.Conv2d(1, 64, 1)

        self.dsn1 = nn.Conv2d(512, 1, 1)
        self.dsn2 = nn.Conv2d(512, 1, 1)
        self.dsn3 = nn.Conv2d(256, 1, 1)
        self.dsn4 = nn.Conv2d(128, 1, 1)
        self.dsn5 = nn.Conv2d(64, 1, 1)

        self.gate5 = GatedSpatialConv2d(64, 64)
        self.gate4 = GatedSpatialConv2d(64, 64)
        self.gate3 = GatedSpatialConv2d(32, 32)
        self.gate2 = GatedSpatialConv2d(16, 16)
        self.gate1 = GatedSpatialConv2d(8, 8)

        self.fuse5 = Fuse(64 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.fuse4 = Fuse(64 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.fuse3 = Fuse(32 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.fuse2 = Fuse(16 + num_filters * 8, num_filters * 2 * 2, num_filters)
        self.fuse1 = Fuse(8 + num_filters, num_filters * 2 * 2, num_filters)

        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)

        self.edge_final = nn.Conv2d(8, 1, kernel_size=1)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)                   #b, 64, h, w
        conv2 = self.conv2(self.pool(conv1))    #b, 128, h/2, w/2
        conv3 = self.conv3(self.pool(conv2))    #b, 256, h/4, w/4
        conv4 = self.conv4(self.pool(conv3))    #b, 512, h/8, w/8
        conv5 = self.conv5(self.pool(conv4))    #b, 512, h/16, w/16

        center = self.center(self.pool(conv5))  #b, 256, h/32, w/32

        x_size = x.size() 
        im_arr = x.cpu().numpy().transpose((0,2,3,1)).astype(np.uint8) 
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3])) #b, 1, h, w
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i],10,100)
        canny = torch.from_numpy(canny).cuda().float()
        edge_inp = F.interpolate(canny, conv5.size()[2:],mode='bilinear', align_corners=True) #b, 1, h/16, w/16
        edge_inp = self.canny_conv(edge_inp) #b, 64, h/16, w/16
        
        d5 = self.dsn5(conv5)
        d4 = self.dsn4(conv4)
        d3 = self.dsn3(conv3)
        d2 = self.dsn2(conv2)
        d1 = self.dsn1(conv1)

        edge5 = self.gate5(edge_inp, d5)
        edge5 = F.interpolate(edge5, conv4.size()[2:], mode='bilinear', align_corners=True) #b, 64, h/8, w/8

        edge4 = self.gate4(edge5, d4)
        edge4 = F.interpolate(edge4, conv3.size()[2:], mode='bilinear', align_corners=True) #b, 32, h/4, w/4

        edge3 = self.gate3(edge4, d3)
        edge3 = F.interpolate(edge3, conv2.size()[2:], mode='bilinear', align_corners=True) #b, 16, h/2, w/2

        edge2 = self.gate2(edge3, d2)
        edge2 = F.interpolate(edge2, conv1.size()[2:], mode='bilinear', align_corners=True) #b, 8, h, w

        edge1 = self.gate1(edge2, d1)
        edge_out = F.sigmoid(self.edge_final(edge1))
        
        dec5 = self.dec5(torch.cat([center, conv5], 1)) #input:256+512, h/32, w/32; output:256, h/16, w/16
        fuse5 = self.fuse5(edge5, dec5)

        dec4 = self.dec4(torch.cat([fuse5, conv4], 1))   #input:256+512, h/16, w/16; output:256, h/8, w/8
        fuse4 = self.fuse4(edge4, dec4)

        dec3 = self.dec3(torch.cat([fuse4, conv3], 1))   #input:256+256, h/8, w/8; output:64, h/4, w/4
        fuse3 = self.fuse3(edge3, dec3)

        dec2 = self.dec2(torch.cat([fuse3, conv2], 1))   #input:64+128, h/4, w/4; output:32, h/2, w/2
        fuse2 = self.fuse2(edge2, dec2)

        dec1 = self.dec1(torch.cat([fuse2, conv1], 1))   #input:32+64, h/2, w/2; output:32, h, w
        fuse1 = self.fuse1(edge1, dec1)

        if self.num_classes > 1:
            seg_out = F.log_softmax(self.final(fuse1), dim=1)
        else:
            seg_out = self.final(fuse1)

        return seg_out, edge_out