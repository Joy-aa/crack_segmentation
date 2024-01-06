import torch
from torch.autograd import Variable
from torchvision.transforms.functional import normalize
import sys
sys.path.append("/home/wj/local/crack_segmentation")
from segtool.data_loader import CrackDataset
import os
import datetime
import tqdm
import numpy as np
from segtool.metric import calc_metric
import cv2
from nets.DSCNet import DSCNet
from modeling.deeplab import DeepLab

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0,1])) 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count())

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def f_score(inputs, target):
    n, c, *_ = inputs.size() #torch.Size([n, 2, 64, 64])
    nt, *_  = target.size() #torch.Size([n, 64, 64])
    #n h c w -> n h w c -> n h*w c
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(nt, -1) #torch.Size([n, h * w])
    
    #--------------------------------------------#
    #   计算f1_score
    #--------------------------------------------#
    
    #agrmax(dim = -1)
    preds = torch.gt(temp_inputs[...,1], temp_inputs[...,0]).int().flatten() #torch.Size([n, h * w])
    
    
    label = temp_target.flatten()
    
    Positive_correct = torch.sum(torch.logical_and(label == 1, preds == 1))
    
    recall = (Positive_correct)  / (label.sum() + 1)
    precision = (Positive_correct)  / (preds.sum() + 1)
    """ 
    计算IOU
    """
    
    return 1, torch.mean(recall), torch.mean(precision)

class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)

def predict(test_loader, model, latest_model_path, save_dir = './result/test_loader', device = torch.device("cuda:0"), vis_sample_id = None):
    if save_dir != '':
        os.makedirs(save_dir, exist_ok=True)

    model.eval()
    pred_list = []
    gt_list = []
    denorm = Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    bar = tqdm.tqdm(total=len(test_loader), ncols=100)
    with torch.no_grad():
        for idx, (img, lab) in enumerate(test_loader, 1):
            val_data  = Variable(img).cuda()
            # val_data = Variable(img).to(device)
            pred = model(val_data)

            image = (denorm(img) * 255).squeeze(0).contiguous().cpu().numpy()
            image = image.transpose(2, 1, 0).astype(np.uint8)
            n, c, h, w = pred.size()
            temp_inputs = torch.softmax(pred.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
            preds = torch.gt(temp_inputs[...,1], temp_inputs[...,0]).int().squeeze(-1)
            mask = preds.squeeze(0).view(h,w).contiguous().cpu().numpy()
            mask = np.expand_dims(mask, axis=0)
            # mask = torch.sigmoid(pred.squeeze(0)).contiguous().cpu().numpy()
            label = lab.contiguous().cpu().numpy()
            
            mask = (mask.transpose(2, 1, 0)*255).astype('uint8')
            label = (label.transpose(2, 1, 0)).astype('uint8')

            mask[mask>127] = 255
            # label[label>0] = 255
            
            pred_list.append(mask)
            gt_list.append(label)

            if idx in vis_sample_id:

                zeros = np.zeros(mask.shape)
                mask = np.concatenate((mask,zeros,zeros),axis=-1).astype(np.uint8)
                label = np.concatenate((zeros,zeros,label),axis=-1).astype(np.uint8)
                
                temp = cv2.addWeighted(label,1,mask,1,0)
                res = cv2.addWeighted(image,0.6,temp,0.4,0)

                cv2.imwrite(os.path.join(save_dir,'%d_test_seg.png' % idx), res)

            bar.update(1)
    bar.close

    metric = calc_metric(pred_list, gt_list, mode='list', threshold=0.5, max_value=255)
    print(metric)
    d = datetime.datetime.today()
    datetime.datetime.strftime(d,'%Y-%m-%d %H-%M-%S')
    os.makedirs('./result_dir', exist_ok=True)
    with open(os.path.join('./result_dir', str(d)+'.txt'), 'a', encoding='utf-8') as fout:
                fout.write(str(latest_model_path)+'\n')
                # for i in range(1, 10): 
                line =  "threshold:0.5 | accuracy:{:.5f} | precision:{:.5f} | recall:{:.5f} | f1:{:.5f} " \
                    .format(metric['accuracy'],  metric['precision'],  metric['recall'],  metric['f1']) + '\n'
                fout.write(line)

if __name__ == '__main__':
    test_dataset = CrackDataset('val',["/mnt/hangzhou_116_homes/DamDetection/data/cutDataset/damV2_overlap0.6_ts1000_slice224"])
    test_loader = torch.utils.data.DataLoader(test_dataset, 1, shuffle=False, pin_memory=False)

    model = DeepLab(backbone='drn', output_stride=16)
    latest_model_path = "/home/wj/local/crack_segmentation/deeplab_drn/checkpoints/deeplabv3+_best_value.pth"
    model.load_state_dict(torch.load(latest_model_path, map_location=device))
    model.to(device)

    # model = DSCNet(3, 2, 15, 1.0, True, device, 16, 1)
    # latest_model_path = "/home/wj/local/crack_segmentation/deeplab_drn/checkpoints/DSCNet_best_value.pth"
    # model.load_state_dict(torch.load(latest_model_path))
    # model.to(device)

    model.eval()

    np.random.seed(0)
    vis_sample_id = np.random.randint(0, len(test_loader), 100, np.int32)  # sample idxs for visualization

    # predict(test_loader, model, latest_model_path, os.path.join(os.getcwd(), 'dsc_test_visual'), device=device, vis_sample_id=vis_sample_id)

    f1 = AverageMeter()
    pbar = tqdm.tqdm(total=len(test_loader),desc='val',postfix=dict,mininterval=0.3)
    for iteration, (data, target) in tqdm.tqdm(enumerate(test_loader)):
        data = Variable(data).cuda()
        output = model(data).cpu()
        _, recall, precision = f_score(output, target / 255)
        f1.update(2 * recall * precision / (recall + precision), 1)
        pbar.set_postfix(**{ 'f1': f1.avg,  })
        # pbar.set_postfix(**{ 'f1': 2 * recall * precision / (recall + precision),  })
        pbar.update(1)
    print(f1.sum, f1.count)