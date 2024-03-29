import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Normal, Independent
from torch.distributions.kl import kl_divergence

import unet_parts as unet_parts

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

class UNet(nn.Module):
    def __init__(self, n_classes, num_blocks = 4):
        super(UNet, self).__init__()
        self.num_blocks = num_blocks
        self.input_block = unet_parts.conv_unit(3, 32, 1).to(device)
        self.down_blocks = []
        self.up_blocks = []

        cur_ch_num = 32
        for _ in range(self.num_blocks):
            self.down_blocks.append(unet_parts.down_block(cur_ch_num, cur_ch_num*2, 3))
            self.up_blocks = [unet_parts.up_block(cur_ch_num*2, cur_ch_num, 3)] + self.up_blocks
            cur_ch_num *= 2

        self.max_channels_num = cur_ch_num
        self.output_block = unet_parts.conv_unit(32, n_classes, 1).to(device)

    def forward(self, x):
        x = self.input_block(x)
        x_inner = [x]

        for i in range(self.num_blocks):
            x = self.down_blocks[i](x)
            x_inner.append(x)

        for i in range(self.num_blocks):
            x = self.up_blocks[i](x, x_inner[-(i+2)])
        x = self.output_block(x)
        return x
    
    
class Conv1x1(nn.Module):
    def __init__(self, n_classes, latent_ch_num, hidden_size = None):
        super(Conv1x1, self).__init__()
        if hidden_size is None:
            hidden_size = n_classes*2
        self.n_classes = n_classes
        self.latent_ch_num = latent_ch_num
        self.conv = nn.Sequential(
            nn.Conv2d(n_classes + latent_ch_num, hidden_size, 1, padding = 0).to(device), 
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, 1, padding = 0).to(device), 
            nn.ReLU(),
            nn.Conv2d(hidden_size, n_classes, 1, padding = 0).to(device)
        )

    def forward(self, unet_res, z):
        bs, _, h,w = unet_res.shape
        z = z.unsqueeze(2).unsqueeze(3).expand(bs, self.latent_ch_num, h, w) #better way?
        inp = torch.cat([unet_res, z], dim=1)
        res = self.conv(inp)
        res = torch.nn.Softmax(dim=1)(res)
        return res
    

class GaussNet(nn.Module):
    def __init__(self, distr_type = "Prior", latent_ch_num = 3, num_blocks = 4, n_classes = 25):
        super(GaussNet, self).__init__()
        
        assert distr_type in {"Prior", "Posterior"}, "Incorrect distribution type"
        if distr_type == "Posterior":
            self.is_posterior = True
            inp_ch_num = n_classes + 3
        else:
            self.is_posterior = False
            inp_ch_num = 3
            
        self.num_blocks = num_blocks
        self.latent_ch_num = latent_ch_num
        
        input_block = unet_parts.conv_unit(inp_ch_num, 32, 3).to(device)
        self.convs = [input_block]
        
        cur_ch_num = 32
        for _ in range(self.num_blocks):
            self.convs.append(unet_parts.down_block(cur_ch_num, cur_ch_num*2, 3))
            cur_ch_num *= 2
            
        self.last_conv = nn.Sequential(
            nn.Conv2d(cur_ch_num, cur_ch_num, 1, padding = 0), 
            nn.ReLU(),
            nn.Conv2d(cur_ch_num, 2*latent_ch_num, 1, padding = 0),
            nn.ReLU()
        ).to(device) 

    def forward(self, x, segm = None):
        bs = x.shape[0]
        
        if self.is_posterior:
            x = torch.cat([segm, x], dim=1)
        
        for i in range(self.num_blocks + 1):
            x = self.convs[i](x)
        
        x = x.mean(-1, keepdim = True).mean(-2, keepdim = True)
        x = self.last_conv(x)
        x = x.squeeze(dim = -1).squeeze(dim = -1)
        
        mu = x[:,:self.latent_ch_num]
        
        log_sigma = x[:,self.latent_ch_num:]
        
        res = []
        for i in range(bs):
            res.append(MultivariateNormal(mu[i], torch.diag(torch.exp(log_sigma[i]))))
        return res
    
    
class ProbUNet(nn.Module):
    def __init__(self, n_classes, latent_ch_num, unet_num_blocks = 4):
        super(ProbUNet, self).__init__()
        self.n_classes =n_classes
        self.unet = UNet(n_classes, unet_num_blocks).to(device)
        self.prior = GaussNet(distr_type="Prior", latent_ch_num = latent_ch_num, n_classes = n_classes).to(device)
        self.posterior = GaussNet(distr_type="Posterior", latent_ch_num = latent_ch_num, n_classes = n_classes).to(device)
        self.combine_layer = Conv1x1(n_classes, latent_ch_num).to(device)
        self.seg_to_one_hot = nn.Embedding(n_classes, n_classes).to(device)
        self.seg_to_one_hot.weight.data = torch.eye(n_classes).to(device)
        self.seg_to_one_hot.weight.requires_grad = False
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight)

    def forward(self, img, seg = None):
        self.unet_res = self.unet(img)
        self.prior_res = self.prior(img)
        if self.training:
            seg = self.seg_to_one_hot(seg.type(torch.cuda.LongTensor)).squeeze(1).permute(0,3,1,2)
            self.post_res = self.posterior(img, seg)
            
    def sample_m(self, img, m):
        """
        Creates m segmentation samples for each item in imgs
        Input:
            img  - torch tensor of size (img_num, 3, h, w)
            m - number of segmentations for each img
        Return:
            segmentation results - tensor of size (img_num, m, 1, h, w)
        """
        self.forward(img)
        img_num, _, h, w = img.shape
        res = torch.ones(img_num, m, self.n_classes, h, w ).cuda()
        z_prior = torch.stack([prior_res_item.sample((m,)) for prior_res_item in self.prior_res])
        for i in range(m):
            #z_prior = torch.stack([prior_res_item.sample() for prior_res_item in self.prior_res])
            #res[:,i] = self.combine_layer(self.unet_res, z_prior)
            res[:,i] = self.combine_layer(self.unet_res, z_prior[:,i])
        return res
            
    def sample(self, img):
        self.forward(img)
        z_prior = torch.stack([prior_res_item.sample() for prior_res_item in self.prior_res])
        return self.combine_layer(self.unet_res, z_prior)
    
    def reconstruct(self, use_posterior_mean = False):
        if use_posterior_mean:
            z_post = torch.stack([post_res_item.mean for post_res_item in self.post_res])
        else:
            z_post = torch.stack([post_res_item.sample() for post_res_item in self.post_res])
        return self.combine_layer(self.unet_res, z_post)
    
    def compute_kl(self):
        return torch.stack([kl_divergence(self.post_res[ind], self.prior_res[ind]) for ind in range(len(self.prior_res))])
    
    def compute_lower_bound(self, imgs, segms, criterion, beta = 1, ignore_index = 255, weight = None):
        self.forward(imgs, segms)
        self.predicted_logits = self.reconstruct()
        #ce_loss = nn.CrossEntropyLoss(weight = weight, reduction = 'mean', ignore_index = ignore_index)
        cross_entropy_loss = criterion(self.predicted_logits, segms, 0).to(device)
        kl = self.compute_kl().mean()
        return cross_entropy_loss + beta * kl