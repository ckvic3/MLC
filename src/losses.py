import mmcv
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ClassWiseLoss(nn.Module):
    def __init__(self, num_classes=80, weights=None):
        super(ClassWiseLoss, self).__init__()
        if weights is not None:
            assert weights.shape[0] == num_classes
        else:
            weights = torch.ones(size=[num_classes])
        self.weights = weights
        self.eps = 1e-4

    def set_weight(self, weights):
        assert max(weights) <= 1.0 and min(weights) >= 0.0
        b = (max(weights) / (min(weights) + self.eps)) - 1.0
        weights = 1 - weights
        p = 2 / (1 + np.exp(-b))
        weights = torch.from_numpy(weights)
        self.weights = torch.pow(weights, p)

        self.weights = F.normalize(self.weights,p=1,dim=0) * 80
        print(self.weights)

    def forward(self, x, y):
        device = (torch.device('cuda')
                  if x.is_cuda
                  else torch.device('cpu'))
        self.weights = self.weights.to(device)
        self.targets = y
        self.anti_targets = 1 - y

        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))
        self.loss *= self.weights
        return -self.loss.mean()


class ReBalancedLoss(nn.Module):
    def __init__(self):
        super(ReBalancedLoss, self).__init__()
        metadata = mmcv.load("/home/pengpeng/MLC/appendix/coco/longtail2017/metadata.pkl")
        class_freq = metadata["class_freq"]
        freq_inv = 1.0 / class_freq
        self.freq_inv = torch.from_numpy(freq_inv).cuda()
        self.map_alpha = 0.1
        self.map_beta = 1.0
        self.map_gamma = 0.2
        self.eps =  1e-4

    def forward(self, x, y):
        weight = self.rebalance_weight(y)
        self.targets = y
        self.anti_targets = 1 - y
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))
        # print(self.loss.shape)
        # print(weight.shape)
        self.loss *= weight

        return -self.loss.mean()


    def rebalance_weight(self, gt_labels):
        # paper formula (2.2)
        repeat_rate = torch.sum(gt_labels.float() * self.freq_inv, dim=1, keepdim=True)
        # paper formula (3)
        pos_weight = self.freq_inv.clone().detach().unsqueeze(0) / repeat_rate
        # paper formula (4) pos and neg are equally treated
        weight = torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha
        return weight

    # def logit_reg_functions(self, labels, logits, weight=None):
    #     if not self.logit_reg:
    #         return logits, weight
    #     if 'init_bias' in self.logit_reg:
    #         logits += self.init_bias
    #     if 'neg_scale' in self.logit_reg:
    #         # paper formula 7
    #         logits = logits * (1 - labels) * self.neg_scale  + logits * labels
    #         weight = weight / self.neg_scale * (1 - labels) + weight * labels
    #     return logits, weight

def rebalance_test():
    metadata = mmcv.load("/home/pengpeng/MLC/appendix/coco/longtail2017/metadata.pkl")
    gt_labels = metadata["gt_labels"]
    gt_labels = np.array(gt_labels)
    class_freq = metadata["class_freq"]
    freq_inv = 1.0 / class_freq

    gt_labels = torch.from_numpy(gt_labels)
    freq_inv = torch.from_numpy(freq_inv)
    repeat_rate = torch.sum(gt_labels.float() * freq_inv, dim=1, keepdim=True)
    print(repeat_rate)
    print(repeat_rate.shape)

    weight = freq_inv / repeat_rate
    print(weight.shape)


class BCELoss(nn.Module):
    def __init__(self) -> None:
        super(BCELoss,self).__init__()
        self.eps = 1e-7

    def forward(self,score,y,mask=None):
        xs_pos = torch.sigmoid(score)
        xs_neg = 1- xs_pos
        xs_pos = torch.clamp(xs_pos,min=self.eps)
        xs_neg = torch.clamp(xs_neg,min=self.eps)

        loss = y * torch.log(xs_pos) + (1-y) * torch.log(xs_neg)
        if mask is not None:
            factor = mask.sum()
            mask = mask.cuda()
            loss = mask * loss
        else:
            factor = y.size(0) * y.size(1)
        loss = torch.neg(loss)
        loss = loss.sum() / factor
        return loss


class BalancedBCELoss(nn.Module):
    def __init__(self,pos_weight,neg_weight) -> None:
        super(BalancedBCELoss,self).__init__()
        self.eps = 1e-7
        self.pos_weight = torch.from_numpy(pos_weight).unsqueeze(0).cuda()
        self.neg_weight = torch.from_numpy(neg_weight).unsqueeze(0).cuda()

    def forward(self,score,y):
        xs_pos = torch.sigmoid(score)
        xs_neg = 1- xs_pos
        xs_pos = torch.clamp(xs_pos,min=self.eps)
        xs_neg = torch.clamp(xs_neg,min=self.eps)

        loss = y * self.pos_weight * torch.log(xs_pos) + (1-y) * self.neg_weight * torch.log(xs_neg)
        loss = torch.neg(loss)
        return torch.mean(loss)




"""
    Most borrow from: https://github.com/Alibaba-MIIL/ASL
"""
import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y, mask):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps, max=1-self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps, max=1-self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    pt0 = xs_pos * y
                    pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
                    pt = pt0 + pt1
                    one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
                    one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            else:
                pt0 = xs_pos * y
                pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
                pt = pt0 + pt1
                one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
                one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w

        if mask is not None:
            mask = mask.cuda()
            loss = mask * loss

        return -loss.mean()

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-5, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y,mask=None):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        factor = 0
        if mask is not None:
            factor = mask.sum()
            mask = mask.cuda()
            self.targets = self.targets * mask
            self.anti_targets = self.anti_targets * mask
        else:
            factor = y.size(0) * y.size(1)
        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(False)
                    self.xs_pos = self.xs_pos * self.targets
                    self.xs_neg = self.xs_neg * self.anti_targets
                    self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                                self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(True)
                self.loss *= self.asymmetric_w
            else:
                self.xs_pos = self.xs_pos * self.targets
                self.xs_neg = self.xs_neg * self.anti_targets
                self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                            self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)   
                self.loss *= self.asymmetric_w         
        _loss = - self.loss.sum() / factor

        return _loss


# class BCELoss(nn.Module):
#     def __init__(self, eps=1e-5):
#         super(BCELoss, self).__init__()
#         self.eps = eps

#     def forward(self, x, y, mask):
#         self.targets = y
#         self.anti_targets = 1 - y
#         self.targets = self.targets * mask
#         self.anti_targets = self.anti_targets * mask

#         x_sigmoid = torch.sigmoid(x)
#         xs_pos = x_sigmoid
#         xs_neg = 1 - x_sigmoid

#         # Basic CE calculation
#         los_pos = y * torch.log(xs_pos.clamp(min=self.eps, max=1 - self.eps))
#         los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps, max=1 - self.eps))
#         loss = los_pos + los_neg

#         return -loss.sum()



if __name__ == '__main__':
    rebalance_test()