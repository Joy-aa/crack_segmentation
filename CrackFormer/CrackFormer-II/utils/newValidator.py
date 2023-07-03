import cv2
import os.path
import torch
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import datetime
class Validator(object):

    def __init__(self, net, valid_log_dir, best_model_dir, normalize = False):

        self.best_model_dir = best_model_dir
        self.valid_log_dir = valid_log_dir + "/valid.txt" # 验证集测试指标的路径
        self.ods = 0
        if os.path.exists(self.best_model_dir)==False:
            os.makedirs(self.best_model_dir)
        self.net = net
        self.normalize = normalize
        # 数值归一化到[-1, 1]
        if self.normalize:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            self.transforms = transforms.ToTensor()

    def get_statistics(self, pred, gt):
        """
        return tp, fp, fn
        """
        tp = np.sum((pred == 1) & (gt == 1))
        fp = np.sum((pred == 1) & (gt == 0))
        fn = np.sum((pred == 0) & (gt == 1))
        return [tp, fp, fn]

    # 计算 ODS 方法
    def cal_prf_metrics(self, pred_list, gt_list, thresh_step=0.01):
        final_accuracy_all = []
        for thresh in np.arange(0.0, 1.0, thresh_step):
            statistics = []

            for pred, gt in zip(pred_list, gt_list):
                
                gt_img = (gt / 255).astype('uint8')
                pred_img = ((pred / 255) > thresh).astype('uint8')
                # calculate each image
                statistics.append(self.get_statistics(pred_img, gt_img))

            # get tp, fp, fn
            tp = np.sum([v[0] for v in statistics])
            fp = np.sum([v[1] for v in statistics])
            fn = np.sum([v[2] for v in statistics])

            # calculate precision
            p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
            # calculate recall
            r_acc = tp / (tp + fn)
            # calculate f-score
            final_accuracy_all.append([thresh, p_acc, r_acc, 2 * p_acc * r_acc / (p_acc + r_acc)])

        return final_accuracy_all

    # 计算 OIS 方法
    def cal_ois_metrics(self,pred_list, gt_list, thresh_step=0.01):
        final_acc_all = []
        for pred, gt in zip(pred_list, gt_list):
            statistics = []
            for thresh in np.arange(0.0, 1.0, thresh_step):
                gt_img = (gt / 255).astype('uint8')
                pred_img = (pred / 255 > thresh).astype('uint8')
                tp, fp, fn = self.get_statistics(pred_img, gt_img)
                p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
                r_acc = tp / (tp + fn)

                if p_acc + r_acc == 0:
                    f1 = 0
                else:
                    f1 = 2 * p_acc * r_acc / (p_acc + r_acc)
                statistics.append([thresh, f1])
            max_f = np.amax(statistics, axis=0)
            final_acc_all.append(max_f[1])
        return np.mean(final_acc_all)

    def validate(self, image):
        self.net.eval()  # 取消掉dropout
        with torch.no_grad():
                x = Variable(self.transforms(image)).cuda()
                x = x.unsqueeze(0)
                outs = self.net.forward(x)  # 前向传播，得到处理后的图像y（tensor形式）
                y = outs[-1]
                img_fused = F.sigmoid(y[0, 0]).data.cpu().numpy()
                # out_clone = output.clone()
                # img_fused = np.squeeze(out_clone.cpu().detach().numpy(), axis=0)

                # img_fused = np.transpose(img_fused, (1, 2, 0))
                
        return img_fused
    
    def calc_metric(self, epoch_num, gt_list, img_list):
        final_results = self.cal_prf_metrics(img_list, gt_list, 0.01)
        final_ois = self.cal_ois_metrics(img_list, gt_list, 0.01)
        max_f = np.amax(final_results, axis=0)
        if max_f[3] > self.ods:
            self.ods = max_f[3]
            self.ois = final_ois
            ods_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-") + str(max_f[3])[0:5]
            print('save ' + ods_str)
            torch.save(self.net.state_dict(), self.best_model_dir + ods_str + ".pth")
        with open(self.valid_log_dir, 'a', encoding='utf-8') as fout:
            line =  "epoch:{} | ODS:{:.6f} | OIS:{:.6f} | max ODS:{:.6f} | max OIS:{:.6f} " \
                .format(epoch_num, max_f[3], final_ois, self.ods, self.ois) + '\n'
            fout.write(line)
        print("epoch={} ODS:{:.6f} | OIS:{:.6f} | max ODS:{:.6f} | max OIS:{:.6f}"
              .format(epoch_num, max_f[3], final_ois, self.ods, self.ois))