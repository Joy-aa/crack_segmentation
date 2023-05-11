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

def cal_precision(y_true, y_pred):
    print(y_true)

if __name__ == "__main__":

    print('aaaa')