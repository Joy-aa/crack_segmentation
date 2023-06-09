from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
import torchvision.transforms as transforms
import torch

def tensorWriter(metric, value, iter):
    writer = SummaryWriter()
    for n_iter in range(100):
        writer.add_scalar('Loss/train', np.random.random(), n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

def getpred(mask, pred, threshold = 0, max_value = 1):
    pred[pred > threshold * max_value] = 1
    pred[pred <= threshold * max_value] = 0
    mask[mask > 0] = 1
    pred_mask = mask
    # pred_mask = pred[:, 0, :, :].contiguous()
    # mask = mask[:, 0, :, :].contiguous()
    return mask, pred_mask

def calc_accuracy(mask, pred_mask):
    mask_acc = pred_mask.eq(mask.view_as(pred_mask)).sum().item() / mask.numel()
    mask_neg_acc = pred_mask[mask < 1].eq(mask[mask < 1].view_as(pred_mask[mask < 1])).sum().item() / mask[mask < 1].numel()
    return mask_acc, mask_neg_acc

def calc_precision(mask, pred_mask):
    num = pred_mask[pred_mask > 0].numel()
    if num == 0:
        precision = 1
    else:
        precision = pred_mask[mask > 0].eq(mask[mask > 0].view_as(pred_mask[mask > 0])).sum().item() / pred_mask[pred_mask > 0].numel()
    return precision

def calc_recall(mask, pred_mask):
    num = mask[mask > 0].numel()
    if num != 0:
            recall = pred_mask[mask > 0].eq(mask[mask > 0].view_as(pred_mask[mask > 0])).sum().item() / mask[mask > 0].numel()
    else:
            recall = 1
    return recall

def calc_f1(mask, pred_mask):
    precision = calc_precision(mask, pred_mask)
    recall = calc_recall(mask, pred_mask)
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2*precision*recall / (precision + recall)
    return f1

def calc_metric(pred_list, gt_list, mode='list', threshold = 0, max_value = 1):
    metric = {
                'accuracy': 0,
                'neg_accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0,}
    if mode == 'list':
        pred_arr = np.array(pred_list)
        gt_arr = np.array(gt_list)
        # print(pred_arr.shape)
        # print(gt_arr.shape)
        th = threshold * max_value
        pred_arr[pred_arr > th] = 1
        pred_arr[pred_arr <= th] = 0
        # print(np.sum(pred_arr == 0))
        # print(np.sum(pred_arr == 1))
        gt_arr[gt_arr > 0] = 1
        # print(np.sum(gt_arr == 0))
        # print(np.sum(gt_arr == 1))
        pred_mask = torch.tensor(pred_arr)
        mask = torch.tensor(gt_arr)
    elif mode == 'tensor':
        # print(pred_list.shape)
        # print(gt_list.shape)
        mask, pred_mask = getpred(pred_list, gt_list, threshold, max_value)

    else:
        print("fault type")
        return metric
    # print(mask[mask<1].numel())
    metric['accuracy'], metric['neg_accuracy'] = calc_accuracy(mask, pred_mask)
    metric['precision'] = calc_precision(mask, pred_mask)
    metric['recall'] = calc_recall(mask, pred_mask)
    metric['f1'] = calc_f1(mask, pred_mask)
    return metric

if __name__ == "__main__":
    from pathlib import Path
    from tqdm import tqdm
    import cv2 as cv
    import os
    DIR_PRED = '/nfs/ymd/result/preprocess'
    DIR_GT = '/nfs/DamDetection/data/new_label'
    paths = [path for path in Path(DIR_PRED).glob('*.*')]
    metrics = {
                'accuracy': 0,
                'neg_accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0,}
    for path in tqdm(paths):
        mask = cv.imread(str(path), 0)
        gt = cv.imread(os.path.join(DIR_GT, path.name), 0)
        metric = calc_metric(mask, gt, 'list')
        metrics['accuracy'] += metric['accuracy'] / len(paths)
        metrics['precision'] += metric['precision'] / len(paths)
        metrics['recall'] += metric['recall'] / len(paths)
        metrics['f1'] += metric['f1'] / len(paths)
    print(metrics)
