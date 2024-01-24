from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
# from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from network.aspp import ASPP
# import mynn
from network.mynn import  Norm2d
import numpy as np
import cv2

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
    def __init__(self, in_channels, middle_channels, out_channels, output_stride):
        super().__init__()
        self.nn = ConvRelu(in_channels, middle_channels)
        self.conv = ConvRelu(middle_channels, out_channels)
        self.aspp = ASPP(inplanes=in_channels, output_stride=output_stride)

    def forward(self, edge_features, texture_features):
        outputs = torch.cat([edge_features, texture_features], 1)
        outputs = self.nn(outputs)
        outputs = self.conv(outputs)
        # outputs = F.interpolate(outputs, scale_factor=2, mode='bilinear')
        return outputs

class _AtrousSpatialPyramidPoolingModule(nn.Module):
    '''
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    '''

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=[6, 12, 18]):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim), nn.ReLU(inplace=True))
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim), nn.ReLU(inplace=True))
         

    def forward(self, x, edge):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, x_size[2:],
                                     mode='bilinear',align_corners=True)
        out = img_features

        edge_features = F.interpolate(edge, x_size[2:],
                                      mode='bilinear',align_corners=True)
        edge_features = self.edge_conv(edge_features)
        out = torch.cat((out, edge_features), 1)

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


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
            torch.nn.BatchNorm2d(in_channels+1),
            nn.Conv2d(in_channels+1, in_channels+1, 1),
            nn.ReLU(), 
            nn.Conv2d(in_channels+1, 1, 1),
            torch.nn.BatchNorm2d(1),
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

class UNetASPP(nn.Module):
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

        self.dsn5 = nn.Conv2d(512, 1, 1)
        self.dsn4 = nn.Conv2d(512, 1, 1)
        self.dsn3 = nn.Conv2d(256, 1, 1)

        self.gate1 = GatedSpatialConv2d(32, 32)
        self.gate2 = GatedSpatialConv2d(16, 16)
        self.gate3 = GatedSpatialConv2d(8, 8)

        self.d1 = nn.Conv2d(64, 32, 1)
        self.d2 = nn.Conv2d(32, 16, 1)
        self.d3 = nn.Conv2d(16, 8, 1)

        self.fuse1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.fuse2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.fuse3 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)

        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)

        self.aspp = _AtrousSpatialPyramidPoolingModule(512, 256, output_stride=16)
        self.bot_aspp = nn.Conv2d(1280 + 256, 512, kernel_size=1, bias=False)

        # self.dec5 = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(512 + num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)

        self.edge_final = nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=False)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)                   #b, 64, h, w
        conv2 = self.conv2(self.pool(conv1))    #b, 128, h/2, w/2
        conv3 = self.conv3(self.pool(conv2))    #b, 256, h/4, w/4
        conv4 = self.conv4(self.pool(conv3))    #b, 512, h/8, w/8
        conv5 = self.conv5(self.pool(conv4))    #b, 512, h/16, w/16

        # center = self.center(self.pool(conv5))  #b, 256, h/32, w/32

        x_size = x.size() 
        im_arr = x.cpu().numpy().transpose((0,2,3,1)).astype(np.uint8) 
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3])) #b, 1, h, w
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i],10,100)
        canny = torch.from_numpy(canny).cuda().float()
        # edge_inp = F.interpolate(canny, conv5.size()[2:],mode='bilinear', align_corners=True) #b, 1, h/16, w/16
        # edge_inp = self.canny_conv(edge_inp) #b, 64, h/16, w/16

        s3 =  F.interpolate(self.dsn3(conv3), x.size()[2:], mode='bilinear', align_corners=True) #b , 1, h/4, w/4 -> b , 1, h, w
        s4 =  F.interpolate(self.dsn4(conv4), x.size()[2:], mode='bilinear', align_corners=True) #b , 1, h/8, w/8 -> b , 1, h, w
        s5 =  F.interpolate(self.dsn5(conv5), x.size()[2:], mode='bilinear', align_corners=True) #b , 1, h/16, w/16 -> b , 1, h, w

        
        cs = self.fuse1(conv1)
        cs = self.d1(cs)
        cs = self.gate1(cs, s3)

        cs = self.fuse2(cs)
        cs = self.d2(cs)
        cs = self.gate2(cs, s4)

        cs = self.fuse3(cs)
        cs = self.d3(cs)
        cs = self.gate3(cs, s5)

        cs = self.edge_final(cs)
        edge_out = torch.sigmoid(cs)
        cat = torch.cat((edge_out, canny), dim=1)
        acts = self.cw(cat)
        acts = torch.sigmoid(acts)

        center = self.aspp(conv5, acts) # b, 1280 + 256, h/16, w/16
        center = self.bot_aspp(center) # b, 256, h/16, w/16

        dec5 = F.interpolate(center, conv4.size()[2:], mode='bilinear', align_corners=True) # b, 512, h/8, w/8
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        seg_out = self.final(dec1)

        return seg_out, edge_out
    
if __name__ == "__main__":
    model = UNetASPP(num_classes=2, pretrained=True, is_deconv=False)
    model.to('cuda:0')
    print(model)
    model.eval()
    input = torch.rand(4, 3, 224, 224).to('cuda:0')
    output = model(input)
    print(output[0].size())
    print(output[1].size())