from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime

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
    correct = np.sum(gt_mask == pred_mask)
    total = gt_mask.shape[0] * gt_mask.shape[1]
    return correct / total

def calc_precision(gt_mask, pred_mask):
    # ts = np.sum(gt_mask == 255)
    tp = np.sum(pred_mask == 255 and gt_mask == pred_mask)
    fp = np.sum(pred_mask == 255 and gt_mask != pred_mask)
    return tp / (tp + fp)

def calc_recall(gt_mask, pred_mask):
    tp = np.sum(pred_mask == 255 and gt_mask == pred_mask)
    fn = np.sum(pred_mask == 0 and gt_mask != pred_mask)
    return tp / (tp + fn)

def call_f1(gt_mask, pred_mask):
    precision = calc_precision(gt_mask, pred_mask)
    recall = calc_recall(gt_mask, pred_mask)
    f1 = 2*precision*recall / (precision + recall)
    

if __name__ == "__main__":
    a = np.random.randint(0, 100, (10, 10))
    b = np.random.randint(0, 100, (10, 10))

    print(np.where(a == b))
    a.shape