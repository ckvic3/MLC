import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import mmcv


class ClsHead(nn.Module):
    """Simplest classification head, with only one fc layer for classification"""

    def __init__(self,
                 in_channels=256,
                 num_classes=80,
                 method='fc'):
        super(ClsHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # self.loss_cls = build_loss(loss_cls)
        self.fc_cls = nn.Linear(in_channels, num_classes)
        self.method = method
        self.no_bias = method in ['matmul', 'cos']

        # self.init_weights()

    def init_weights(self):
        print("cls head init...")
        nn.init.normal_(self.fc_cls.weight,0,0.1)
        nn.init.constant_(self.fc_cls.bias, 0)
        if self.no_bias:
            self.fc_cls.bias.requires_grad = False
            print('No bias for classifier!')

    def forward(self, x):
        x = x.view(x.size(0), -1)
        if self.method in ['fc', 'matmul']:
            cls_score = self.fc_cls(x)
        elif self.method == 'cos':
            cls_score = torch.cosine_similarity(x.unsqueeze(-1), self.fc_cls.weight.t().unsqueeze(0), dim=1)
        elif self.method == 'no_zero':
            cls_score = self.nonzero_cosine_similarity(x, self.fc_cls.weight)
        elif self.method == 'norm_w':
            cls_weight = self.fc_cls.weight / torch.norm(self.fc_cls.weight, 2, 1, keepdim=True)
            cls_score = torch.mm(x, cls_weight.t()) + self.fc_cls.bias
        else:
            raise NameError
        return cls_score


    # def loss(self,
    #          cls_score,
    #          labels,
    #          masks,
    #          weight=None,
    #          reduction_override=None):
    #     losses = dict()
    #     losses['loss_cls'] = self.loss_cls(
    #         cls_score,
    #         labels,
    #         masks,
    #         weight,
    #         avg_factor=None,
    #         reduction_override=reduction_override)
    #     if isinstance(cls_score, list):
    #         cls_score = cls_score[0]
    #     losses['acc'] = accuracy(cls_score, labels)
    #     return losses
    #
    # def nonzero_cosine_similarity(self, x1, x2):
    #     # mask = x1 != 0
    #     cls_score = torch.zeros(x1.shape[0], self.num_classes).cuda()
    #     for i, x in enumerate(x1):
    #         mask = (x != 0).float()
    #         # print(torch.sum(mask))
    #         cls_score[i] = torch.cosine_similarity(x.unsqueeze(0),x2*mask)
    #
    #     return cls_score


class FCHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden=256):
        super(FCHead, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_features=in_channels, out_features=256),
                                nn.ReLU(),
                                nn.Linear(in_features=256, out_features=num_classes))

        self._init_weight()

    def _init_weight(self):
        for m in self.fc.children():
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.fc(x)
        return out
