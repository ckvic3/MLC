import torch
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from copy import deepcopy
from src.models.pfc import PFC
from src.models.cls_head import ClsHead, FCHead
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, num_classes=80, pretrained=True, frozen_stages=1):
        super(BaseModel, self).__init__()
        self.return_layers = {"avgpool": "0", "layer4": "1", }
        self.backbone = IntermediateLayerGetter(resnet50(pretrained=pretrained), self.return_layers)
        self.neck = PFC(in_channels=2048, out_channels=256, dropout=0.5)
        self.fc = ClsHead(in_channels=256, num_classes=num_classes)
        
        self.frozen_stages = frozen_stages
        # self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.backbone.bn1.eval()
            for m in [self.backbone.conv1, self.backbone.bn1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self.backbone, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)['0'].view(x.shape[0], -1)
        x = self.neck(x)
        x = self.fc(x)
        return x


# class NormModel(nn.Module):
#     def __init__(self, num_classes=80, frozen_stages=0):
#         super(NormModel, self).__init__()
#         self.backbone = resnet50(pretrained=True)
#         # self.head = FCHead(in_channels=2048,num_classes=num_classes)
#         self.backbone.fc = nn.Linear(in_features=2048, out_features=num_classes)
#         self.frozen_stages = frozen_stages
#         self._freeze_stages()

#     def _freeze_stages(self):
#         if self.frozen_stages >= 0:
#             self.backbone.bn1.eval()
#             for m in [self.backbone.conv1, self.backbone.bn1]:
#                 for param in m.parameters():
#                     param.requires_grad = False

#         for i in range(1, self.frozen_stages + 1):
#             m = getattr(self.backbone, 'layer{}'.format(i))
#             m.eval()
#             for param in m.parameters():
#                 param.requires_grad = False

#     def forward(self, x):
#         out = self.backbone(x)
#         return out

class NormModel(nn.Module):
    def __init__(self, num_classes=80, frozen_stages=1):
        super(NormModel, self).__init__()
        self.return_layers = {"avgpool": "0", "layer4": "1", }
        self.backbone = IntermediateLayerGetter(
            resnet50(pretrained=True), self.return_layers)
        # self.head = FCHead(in_channels=2048,num_classes=num_classes)
        # self.head = nn.Linear(in_features=2048, out_features=num_classes)
        self.head = nn.Sequential(
            PFC(in_channels=2048,out_channels=256, dropout=0.5),
            ClsHead(in_channels=256,num_classes=num_classes)
        )                          
        self.frozen_stages = frozen_stages
        # self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.backbone.bn1.eval()
            for m in [self.backbone.conv1, self.backbone.bn1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self.backbone, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        fea = self.backbone(x)['0']
        fea = fea.view(fea.shape[0], -1)
        out = self.head(fea)
        return out



class MultiStageModel(nn.Module):
    def __init__(self, groups):
        super(MultiStageModel, self).__init__()
        self.return_layers = {"avgpool": "0", "layer4": "1", }
        self.backbone = IntermediateLayerGetter(resnet50(pretrained=True), self.return_layers)
        self.fcs = nn.ModuleList()
        self.stage = 0
        self.groups = groups
        for i in range(len(groups)):
            # self.fcs.add_module(name=str(i), module=nn.Linear(in_features=2048, out_features=len(groups[i])))
            self.fcs.add_module(name=str(i),
                                module=nn.Sequential(
                                    PFC(in_channels=2048, out_channels=256, dropout=0.5),
                                    ClsHead(in_channels=256, num_classes=len(groups[i]))))

        self.policy = None
        # self._init_params()

    def _init_params(self):
        for m in self.fcs.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def set_stage(self, stage):
        self.stage = stage

    def set_policy(self, policy: str):
        self.policy = policy

    def forward(self, x):
        assert self.stage <= len(self.groups)
        feas = self.backbone(x)
        f = feas['0'].view(x.shape[0], -1)
        if self.training:
            out = self.fcs[self.stage](f)
        else:  # 测试时，需要把所有的结果都输出
            out = []
            for i in range(len(self.groups)):
                out.append(self.fcs[i](f))
        return out, feas


class FreezeModel(nn.Module):
    def __init__(self, model, device=None):
        super(FreezeModel, self).__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)




def model_test():
    model = NormModel()
    for name, params in model.named_parameters(recurse=True):
        print(name, params.requires_grad)


if __name__ == '__main__':
    model_test()
