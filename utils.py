from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch

def tensorWriter(metric, value, iter):
    writer = SummaryWriter()
    for n_iter in range(100):
        writer.add_scalar('Loss/train', np.random.random(), n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

def getmask(prob_img, threshold = 0):
    mask = np.zeros(prob_img.shape, dtype='uint8')
    img=(prob_img * 255).astype(np.uint8)
    mask[img > threshold] = 255
    return mask

def calc_accuracy(gt_mask, pred_mask):
    gt_mask_fla = gt_mask.flatten()
    print(gt_mask_fla.shape)
    pred_mask_fla = pred_mask.flatten()
    correct = np.sum(gt_mask_fla == pred_mask_fla)
    total = gt_mask_fla.shape[0]
    # print(gt_mask_fla.shape)
    return correct / total

def calc_precision(gt_mask, pred_mask):
    gt_mask_fla = gt_mask.flatten()
    pred_mask_fla = pred_mask.flatten()
    tind = np.where(pred_mask_fla == 255)
    tp = np.sum(pred_mask_fla[tind] == gt_mask_fla[tind])
    fp = np.sum(pred_mask_fla[tind] != gt_mask_fla[tind])
    total = tp+fp
    if total == 0:
        total = 0x3f3f3f3f
    return tp / total

def calc_recall(gt_mask, pred_mask):
    gt_mask_fla = gt_mask.flatten()
    pred_mask_fla = pred_mask.flatten()
    tind = np.where(pred_mask_fla == 255)
    find = np.where(pred_mask_fla == 0)
    tp = np.sum(pred_mask_fla[tind] == gt_mask_fla[tind])
    fn = np.sum(pred_mask_fla[find] != gt_mask_fla[find])
    total = tp+fn
    if total == 0:
        total = 0x3f3f3f3f
    return tp / total

def call_f1(gt_mask, pred_mask):
    precision = calc_precision(gt_mask, pred_mask)
    recall = calc_recall(gt_mask, pred_mask)
    f1 = 2*precision*recall / (precision + recall)
    
def miou(pred: torch.Tensor, mask: torch.Tensor):
    pred, mask = pred.flatten().long(), mask.flatten().long()
    correct = torch.bincount(mask[pred == mask], minlength=2)
    pred_count = torch.bincount(pred, minlength=2)
    total = torch.bincount(mask, minlength=2)
    union = pred_count + total - correct
    union[union == 0] = 1
    iou = correct / union
    return iou.mean()

def calc_miou(gt_mask, pred_mask):
    mask = gt_mask.flatten()
    pred = pred_mask.flatten()
    correct = np.bincount(mask[pred == mask], minlength=256)
    pred_count = np.bincount(pred, minlength=256)
    total = np.bincount(mask, minlength=256)
    union = pred_count + total - correct
    union[union == 0] = 1
    iou = correct / union
    return iou.mean()

if __name__ == "__main__":
    a = np.random.randint(0, 100, (4, 1, 10, 10))
    b = np.random.randint(0, 100, (4, 1, 10, 10))
    # a = np.zeros((4,1,10,10))
    # b = np.ones((4,1,10,10)) * 255
    # a[1,0, 4,4] = 255
    # a[1,0, 4,5] = 255
    # b[1,0, 4,5] = 0
    # print(a)
    # print(b)
    print(calc_miou(a,b))
    # a.shape