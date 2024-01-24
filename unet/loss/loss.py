"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
# from config import cfg
from loss.DualTaskLoss import DualTaskLoss
from torch import Tensor

def get_loss(args):
    '''
    Get the criterion based on the loss function
    args: 
    return: criterion
    '''
    if args.joint_edgeseg_loss:
        criterion = JointEdgeSegLoss(classes=2,ignore_index=255, upper_bound=1,edge_weight=1, seg_weight=1, att_weight=1, dual_weight=1).cuda()

        criterion_val = JointEdgeSegLoss(classes=2, mode='val',ignore_index=255, upper_bound=1,edge_weight=1, seg_weight=1, att_weight=0, dual_weight=0).cuda()
    else:
        criterion = SegEdgeSegLoss(classes=2, ignore_index=255, dice=args.r0, normalise=args.normal, seg_ce_weight=args.r1, edge_weight=args.r2, threshold=args.att_th).cuda()
        criterion_val = SegEdgeSegLoss(classes=2, ignore_index=255, dice=False, seg_ce_weight=1, edge_weight=20).cuda()
                               
    return criterion, criterion_val

class JointEdgeSegLoss(nn.Module):
    def __init__(self, classes, weight=None, reduction='mean', ignore_index=255,
                 norm=False, upper_bound=1.0, mode='train', 
                 edge_weight=1, seg_weight=1, att_weight=1, dual_weight=1, edge='none'):
        super(JointEdgeSegLoss, self).__init__()
        self.num_classes = classes
        if mode == 'train':
            self.seg_loss = ImageBasedCrossEntropyLoss2d(
                    classes=classes, ignore_index=ignore_index, upper_bound=upper_bound).cuda()
        elif mode == 'val':
            self.seg_loss = CrossEntropyLoss2d(size_average=True,
                                               ignore_index=ignore_index).cuda()

        self.edge_weight = edge_weight
        self.seg_weight = seg_weight
        self.att_weight = att_weight
        self.dual_weight = dual_weight

        self.dual_task = DualTaskLoss()

    def bce2d(self, input, target):
        n, c, h, w = input.size()
    
        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()

        pos_index = (target_t ==1)
        neg_index = (target_t ==0)
        ignore_index=(target_t >1)

        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ignore_index=ignore_index.data.cpu().numpy().astype(bool)

        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num*1.0 / sum_num
        weight[neg_index] = pos_num*1.0 / sum_num

        weight[ignore_index] = 0

        weight = torch.from_numpy(weight)
        weight = weight.cuda()
        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, size_average=True)
        return loss

    def edge_attention(self, input, target, edge):
        n, c, h, w = input.size()
        filler = torch.ones_like(target) * 255
        return self.seg_loss(input, 
                             torch.where(edge.max(1)[0] > 0.8, target, filler))

    def forward(self, inputs, targets):
        segin, edgein = inputs
        segmask, edgemask = targets

        losses = {}

        losses['seg_loss'] = self.seg_weight * self.seg_loss(segin, segmask)
        losses['edge_loss'] = self.edge_weight * 20 * self.bce2d(edgein, edgemask)
        attention_edge = self.edge_attention(segin, segmask, edgein)
        if(torch.isnan(attention_edge)):
            attention_edge = torch.ones_like(attention_edge)
        losses['att_loss'] = self.att_weight * attention_edge
        # losses['att_loss'] = self.att_weight * self.edge_attention(segin, segmask, edgein)
        losses['dual_loss'] = self.dual_weight * self.dual_task(segin, segmask)
              
        return losses
    
class SegEdgeSegLoss(nn.Module):
    def __init__(self, classes, ignore_index=255, dice = True, normalise = False, seg_ce_weight=5, edge_weight=10, threshold=0.8):
        super(SegEdgeSegLoss, self).__init__()
        self.num_classes = classes
        self.th = threshold
        # if mode == 'train':
        #     self.seg_loss = ImageBasedCrossEntropyLoss2d(
        #             classes=classes, ignore_index=ignore_index, upper_bound=upper_bound).cuda()
        # elif mode == 'val':
        self.seg_loss = CrossEntropyLoss2d(size_average=True, ignore_index=ignore_index).cuda()
        self.dice_loss = DiceLoss(multiclass=False)
        self.dual_task = DualTaskLoss()

        self.dice = dice
        self.normalise = normalise
        self.edge_weight = edge_weight
        self.seg_weight = seg_ce_weight

    def bce2d(self, input, target):
        n, c, h, w = input.size()
    
        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()

        pos_index = (target_t ==1)
        neg_index = (target_t ==0)
        ignore_index=(target_t >1)

        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ignore_index=ignore_index.data.cpu().numpy().astype(bool)

        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num*1.0 / sum_num
        weight[neg_index] = pos_num*1.0 / sum_num

        weight[ignore_index] = 0

        weight = torch.from_numpy(weight)
        weight = weight.cuda()
        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, size_average=True)
        return loss

    def edge_attention(self, input, target, edge):
        n, c, h, w = input.size()
        filler = torch.ones_like(target) * 255
        return self.seg_loss(input, 
                             torch.where(edge.max(1)[0] > self.th, target, filler))

    def forward(self, inputs, targets):
        segin, edgein = inputs
        segmask, edgemask = targets

        losses = {}

        losses['seg_loss'] = self.seg_weight * self.seg_loss(segin, segmask)
        losses['edge_loss'] = self.edge_weight * self.bce2d(edgein, edgemask)
        if self.dice:
            losses['dice_loss'] = self.dice_loss(segin, segmask)
        
        if self.normalise:
            attention_edge = self.edge_attention(segin, segmask, edgein)
            if(torch.isnan(attention_edge)):
                attention_edge = torch.zeros_like(attention_edge)
            losses['att_loss'] =  attention_edge
            # losses['att_loss'] = self.att_weight * self.edge_attention(segin, segmask, edgein)
            losses['dual_loss'] = self.dual_task(segin, segmask)
              
        return losses

#Img Weighted Loss
class ImageBasedCrossEntropyLoss2d(nn.Module):

    def __init__(self, classes, weight=None, size_average=True, ignore_index=255,
                 norm=False, upper_bound=1.0):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        logging.info("Using Per Image based weighted loss")
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = False

    def calculateWeights(self, target):
        hist = np.histogram(target.flatten(), range(
            self.num_classes + 1), density=True)[0]
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist

    def forward(self, inputs, targets):
        target_cpu = targets.data.cpu().numpy()
        if self.batch_weights:
            weights = self.calculateWeights(target_cpu)
            self.nll_loss.weight = torch.Tensor(weights).cuda()

        loss = 0.0
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                weights = self.calculateWeights(target_cpu[i])
                self.nll_loss.weight = torch.Tensor(weights).cuda()
            
            loss += self.nll_loss(F.log_softmax(inputs[i].unsqueeze(0), dim=1),
                                          targets[i].unsqueeze(0))
        return loss


#Cross Entroply NLL Loss
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        logging.info("Using Cross Entropy Loss")
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)
    


class DiceLoss(nn.Module):
    def __init__(self, multiclass: bool = False, reduce_batch_first: bool = False, epsilon: float = 1e-6):
        super(DiceLoss, self).__init__()
        logging.info("Using Dice Loss")
        self.epsilon = epsilon
        self.reduce_batch_first = reduce_batch_first
        self.multiclass = multiclass

    def calc_loss(self, inputs: Tensor, target: Tensor, beta=1, smooth = 1e-5):
        n, c, h, w = inputs.size()
        # nt, ht, wt, ct = target.size()
        # if h != ht and w != wt:
        #     inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
            
        temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
        temp_target = target.view(n, -1, c)
    
        #--------------------------------------------#
        #   计算dice loss
        #--------------------------------------------#
        tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
        fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
        fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp
    
        score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        dice_loss = 1 - torch.mean(score)
        return dice_loss

    
    def dice_coeff(self, input: Tensor, target: Tensor):
        # Average of Dice coefficient for all batches, or for a single mask
        assert input.size() == target.size()
        # print(input.dim())
        assert input.dim() == 3 or not self.reduce_batch_first

        sum_dim = (-1, -2) if input.dim() == 2 or not self.reduce_batch_first else (-1, -2, -3)

        inter = 2 * (input * target).sum(dim=sum_dim)
        sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
        sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

        dice = (inter + self.epsilon) / (sets_sum + self.epsilon)
        return dice.mean()


    def multiclass_dice_coeff(self, input: Tensor, target: Tensor):
        # Average of Dice coefficient for all classes
        return self.dice_coeff(input.flatten(0, 1), target.flatten(0, 1), self.reduce_batch_first, self.epsilon)


    def forward(self, input: Tensor, target: Tensor):
        n, c, h, w = input.size()
        temp_inputs = torch.softmax(input.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
        preds = torch.gt(temp_inputs[...,1], temp_inputs[...,0]).int().squeeze(-1)
        # preds = torch.argmax(temp_inputs, dim=-1)

        # target = target.view(n, -1)
        mask = preds.view(n, h, w)
        # Dice loss (objective to minimize) between 0 and 1
        fn = self.multiclass_dice_coeff if self.multiclass else self.dice_coeff
        return 1 - fn(mask, target)
        # return self.calc_loss(input, target)

# self.seg_loss = DiceLoss(multiclass=False)

