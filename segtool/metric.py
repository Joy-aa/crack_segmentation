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
    # pred_mask = pred[:, 0, :, :].contiguous()
    # mask = mask[:, 0, :, :].contiguous()
    return mask, pred

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

def calc_f1(mask, pred_mask, tmp_precision = None, tmp_recall = None):
    if tmp_precision is not None and tmp_recall is not None:
        precision = tmp_precision
        recall = tmp_recall
    else:
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
        mask_arr = pred_arr.copy()
        mask_arr[pred_arr > th] = 1
        mask_arr[pred_arr <= th] = 0
        # print(np.sum(pred_arr == 0))
        # print(np.sum(pred_arr == 1))
        gt_arr[gt_arr > 0] = 1
        # print(np.sum(gt_arr == 0))
        # print(np.sum(gt_arr == 1))
        pred_mask = torch.tensor(mask_arr)
        mask = torch.tensor(gt_arr)
    elif mode == 'tensor':
        # print(pred_list.shape)
        # print(gt_list.shape)
        mask, pred_mask = getpred(gt_list, pred_list, threshold, max_value)

    else:
        print("fault type")
        return metric
    # print(mask[mask<1].numel())
    metric['accuracy'], metric['neg_accuracy'] = calc_accuracy(mask, pred_mask)
    metric['precision'] = calc_precision(mask, pred_mask)
    metric['recall'] = calc_recall(mask, pred_mask)
    metric['f1'] = calc_f1(mask, pred_mask, metric['precision'], metric['recall'])
    return metric

if __name__ == "__main__":
    from pathlib import Path
    from tqdm import tqdm
    import cv2 as cv
    import os
    DIR_PRED = '/home/wj/local/crack_segmentation/unet++/result_images'
    DIR_GT = '/mnt/nfs/wj/data/new_label'
    paths = [path for path in Path(DIR_PRED).glob('*.*')]
    metrics=[]
    for path in tqdm(paths):
        print(str(path))
        pred_list = []
        gt_list = []
        mask = cv.imread(str(path), 0)
        mask = mask / 255.0
        gt = cv.imread(os.path.join(DIR_GT, path.stem+'.png'), 0)

        filepath = os.path.join('/mnt/nfs/wj/result-stride_0.7/Jun02_06_33_42/box', path.stem+'.txt')
        boxes = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for data in f.readlines():
                box = data.split(' ')[:-1]
                boxes.append(box)
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            img_pat = mask[y1:y2,x1:x2]
            gt_pat = gt[y1:y2, x1:x2]
            ori_shape = gt_pat.shape
            img_pat = cv.resize(img_pat, (128, 128), cv.INTER_AREA)
            gt_pat = cv.resize(gt_pat, (128, 128), cv.INTER_AREA)
            pred_list.append(img_pat)
            gt_list.append(gt_pat)
        
        for i in range(1, 10):
                    threshold = i / 10
                    metric = calc_metric(pred_list, gt_list, mode='list', threshold=threshold)
                    print(metric)
                    metric['accuracy'] = metric['accuracy'] / len(paths)
                    metric['precision'] = metric['precision'] / len(paths)
                    metric['recall'] = metric['recall'] / len(paths)
                    metric['f1'] = metric['f1'] / len(paths)
                    if len(metrics) < i:
                        metrics.append(metric)
                    else:
                        metrics[i-1]['accuracy'] += metric['accuracy']
                        metrics[i-1]['precision'] += metric['precision']
                        metrics[i-1]['recall'] += metric['recall']
                        metrics[i-1]['f1'] += metric['f1']
    print(metrics)
    d = datetime.today()
    datetime.strftime(d,'%Y-%m-%d %H-%M-%S')
    os.makedirs('./result_dir', exist_ok=True)
    with open(os.path.join('./result_dir', str(d)+'.txt'), 'a', encoding='utf-8') as fout:
            fout.write(DIR_PRED+'\n')
            for i in range(1, 10): 
                    line =  "threshold:{:d} | accuracy:{:.5f} | precision:{:.5f} | recall:{:.5f} | f1:{:.5f} " \
                        .format(i, metrics[i-1]['accuracy'],  metrics[i-1]['precision'],  metrics[i-1]['recall'],  metrics[i-1]['f1']) + '\n'
                    fout.write(line)
