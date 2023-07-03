from torch import nn
import torch
import torch.nn.functional as F
import sys
sys.path.append("/home/wj/local/crack_segmentation/DeepCrack/codes")
from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from config import Config as cfg
from model.deepcrack import DeepCrack
from trainer import DeepCrackTrainer


def Conv1X1(in_, out):
    return torch.nn.Conv2d(in_, out, 1, padding=0)


def Conv3X3(in_, out):
    return torch.nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = Conv3X3(in_, out)
        # self.bn = SynchronizedBatchNorm2d(out)
        self.activation = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        x = self.activation(x)
        return x

class Down(nn.Module):

    def __init__(self, nn):
        super(Down,self).__init__()
        self.nn = nn
        self.maxpool_with_argmax = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self,inputs):
        down = self.nn(inputs)
        unpooled_shape = down.size()
        outputs, indices = self.maxpool_with_argmax(down)
        return outputs, down, indices, unpooled_shape

class Up(nn.Module):

    def __init__(self, nn):
        super().__init__()
        self.nn = nn
        self.unpool=torch.nn.MaxUnpool2d(kernel_size=2, stride=2)

    def forward(self,inputs,indices,output_shape):
        outputs = self.unpool(inputs, indices=indices, output_size=output_shape)
        outputs = self.nn(outputs)
        return outputs

class Fuse(nn.Module):

    def __init__(self, nn, scale):
        super().__init__()
        self.nn = nn
        self.scale = scale
        self.conv = Conv1X1(64,1)

    def forward(self,down_inp,up_inp):
        outputs = torch.cat([down_inp, up_inp], 1)
        outputs = self.nn(outputs)
        outputs = self.conv(outputs)
        outputs = F.interpolate(outputs, scale_factor=self.scale, mode='bilinear')
        return outputs



class DeepCrackV2(nn.Module):

    def __init__(self, pretrained_model = False):
        super(DeepCrackV2, self).__init__()

        self.premodel = DeepCrack()
        trainer = DeepCrackTrainer(self.premodel)
        if pretrained_model:
            pretrained_dict = trainer.saver.load(cfg.pretrained_model, multi_gpu=True)
            premodel_dict = self.premodel.state_dict()

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in premodel_dict}
            premodel_dict.update(pretrained_dict)
            self.premodel.load_state_dict(premodel_dict)

        # self.bn = SynchronizedBatchNorm2d()
        self.relu = nn.ReLU(inplace=True)

        self.down1 = Down(nn.Sequential(
            self.premodel.down1.nn[0].conv,
            SynchronizedBatchNorm2d(64),
            self.relu,
            self.premodel.down1.nn[1].conv,
            SynchronizedBatchNorm2d(64),
            self.relu,
        ))

        self.down2 = Down(nn.Sequential(
            self.premodel.down2.nn[0].conv,
            SynchronizedBatchNorm2d(128),
            self.relu,
            self.premodel.down2.nn[1].conv,
            SynchronizedBatchNorm2d(128),
            self.relu,
        ))

        self.down3 = Down(nn.Sequential(
            self.premodel.down3.nn[0].conv,
            SynchronizedBatchNorm2d(256),
            self.relu,
            self.premodel.down3.nn[1].conv,
            SynchronizedBatchNorm2d(256),
            self.relu,
            self.premodel.down3.nn[2].conv,
            SynchronizedBatchNorm2d(256),
            self.relu,
        ))

        self.down4 = Down(nn.Sequential(
            self.premodel.down4.nn[0].conv,
            SynchronizedBatchNorm2d(512),
            self.relu,
            self.premodel.down4.nn[1].conv,
            SynchronizedBatchNorm2d(512),
            self.relu,
            self.premodel.down4.nn[2].conv,
            SynchronizedBatchNorm2d(512),
            self.relu,
        ))

        self.down5 = Down(nn.Sequential(
            self.premodel.down5.nn[0].conv,
            SynchronizedBatchNorm2d(512),
            self.relu,
            self.premodel.down5.nn[1].conv,
            SynchronizedBatchNorm2d(512),
            self.relu,
            self.premodel.down5.nn[2].conv,
            SynchronizedBatchNorm2d(512),
            self.relu,
        ))

        self.up1 = Up(nn.Sequential(
            self.premodel.up1.nn[0].conv,
            SynchronizedBatchNorm2d(64),
            self.relu,
            self.premodel.up1.nn[1].conv,
            SynchronizedBatchNorm2d(64),
            self.relu,
        ))

        self.up2 = Up(nn.Sequential(
            self.premodel.up2.nn[0].conv,
            SynchronizedBatchNorm2d(128),
            self.relu,
            self.premodel.up2.nn[1].conv,
            SynchronizedBatchNorm2d(64),
            self.relu,
        ))

        self.up3 = Up(nn.Sequential(
            self.premodel.up3.nn[0].conv,
            SynchronizedBatchNorm2d(256),
            self.relu,
            self.premodel.up3.nn[1].conv,
            SynchronizedBatchNorm2d(256),
            self.relu,
            self.premodel.up3.nn[2].conv,
            SynchronizedBatchNorm2d(128),
            self.relu,
        ))

        self.up4 = Up(nn.Sequential(
            self.premodel.up4.nn[0].conv,
            SynchronizedBatchNorm2d(512),
            self.relu,
            self.premodel.up4.nn[1].conv,
            SynchronizedBatchNorm2d(512),
            self.relu,
            self.premodel.up4.nn[2].conv,
            SynchronizedBatchNorm2d(256),
            self.relu,
        ))

        self.up5 = Up(nn.Sequential(
            self.premodel.up5.nn[0].conv,
            SynchronizedBatchNorm2d(512),
            self.relu,
            self.premodel.up5.nn[1].conv,
            SynchronizedBatchNorm2d(512),
            self.relu,
            self.premodel.up5.nn[2].conv,
            SynchronizedBatchNorm2d(512),
            self.relu,
        ))

        self.fuse5 = Fuse(self.premodel.fuse5.nn, scale = 16)
        self.fuse4 = Fuse(self.premodel.fuse4.nn, scale = 8)
        self.fuse3 = Fuse(self.premodel.fuse3.nn, scale = 4)
        self.fuse2 = Fuse(self.premodel.fuse2.nn, scale = 2)
        self.fuse1 = Fuse(self.premodel.fuse1.nn, scale = 1)

        self.final = Conv1X1(5, 1)

    def forward(self,inputs):

        # encoder part
        out, down1, indices_1, unpool_shape1 = self.down1(inputs)
        out, down2, indices_2, unpool_shape2 = self.down2(out)
        out, down3, indices_3, unpool_shape3 = self.down3(out)
        out, down4, indices_4, unpool_shape4 = self.down4(out)
        out, down5, indices_5, unpool_shape5 = self.down5(out)
        # print(down1.shape, down2.shape, down3.shape, down4.shape, down5.shape)

        # decoder part
        up5 = self.up5(out, indices=indices_5, output_shape=unpool_shape5)
        up4 = self.up4(up5, indices=indices_4, output_shape=unpool_shape4)
        up3 = self.up3(up4, indices=indices_3, output_shape=unpool_shape3)
        up2 = self.up2(up3, indices=indices_2, output_shape=unpool_shape2)
        up1 = self.up1(up2, indices=indices_1, output_shape=unpool_shape1)
        # print(up1.shape, up2.shape, up3.shape, up4.shape, up5.shape)

        fuse5 = self.fuse5(down_inp=down5, up_inp=up5)
        fuse4 = self.fuse4(down_inp=down4, up_inp=up4)
        fuse3 = self.fuse3(down_inp=down3, up_inp=up3)
        fuse2 = self.fuse2(down_inp=down2, up_inp=up2)
        fuse1 = self.fuse1(down_inp=down1, up_inp=up1)
        # print(fuse1.shape, fuse2.shape, fuse3.shape, fuse4.shape, fuse5.shape)

        output = self.final(torch.cat([fuse5,fuse4,fuse3,fuse2,fuse1],1))

        return output, fuse5, fuse4, fuse3, fuse2, fuse1

if __name__ == '__main__':
    premodel = DeepCrack()
    trainer = DeepCrackTrainer(premodel)
    if cfg.pretrained_model:
        pretrained_dict = trainer.saver.load(cfg.pretrained_model, multi_gpu=True)
        premodel_dict = premodel.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in premodel_dict}
        premodel_dict.update(pretrained_dict)
        premodel.load_state_dict(premodel_dict)
        trainer.vis.log('load checkpoint: %s' % cfg.pretrained_model, 'train info')
    premodel1 = DeepCrackV2(cfg.pretrained_model)
    print(premodel1)

    # out = premodel(inp)

