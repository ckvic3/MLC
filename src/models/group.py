import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet
from torch.nn import functional as F
from .backbone import CreateBackbone,Bottleneck,BasicBlock
from .pfc import PFC
from .cls_head import ClsHead
import time
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import resnet50
from fightingcv_attention.attention.CBAM import CBAMBlock

expansions = {"resnet18": 512,
              "resnet50": 2048,
              "resnet101": 2048}


class BaseModel(nn.Module):
    def __init__(self,num_classes=80) -> None:
        super(BaseModel,self).__init__()
        self.backbone = IntermediateLayerGetter(resnet50(pretrained=True),return_layers={"avgpool":"avgpool"})
        self.neck = PFC(in_channels=2048,out_channels=256,dropout=0.5)
        self.fc = ClsHead(in_channels=256, num_classes=num_classes)
        
    def forward(self,x):
        x = self.backbone(x)["avgpool"]
        x = x.view(x.shape[0],-1)
        x = self.neck(x)
        x = self.fc(x)
        return x

class CBAMModel(nn.Module):
    def __init__(self,num_classes=80) -> None:
        super(CBAMModel,self).__init__()
        self.backbone = IntermediateLayerGetter(resnet50(pretrained=True),return_layers={"layer4":"layer4"})
        self.attention = CBAMBlock(channel=2048,reduction=16,kernel_size=7)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.neck = PFC(in_channels=2048,out_channels=256,dropout=0.5)
        self.fc = ClsHead(in_channels=256, num_classes=num_classes)
        
    def forward(self,x):
        x = self.backbone(x)["layer4"]       
        x = self.attention(x)
        x = self.avgpool(x)    
        x = x.view(x.shape[0],-1)
        x = self.neck(x)
        x = self.fc(x)
        return x




def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class GroupModel(nn.Module):
    def __init__(self, super_labels, arch: str = "resnet50", block=Bottleneck, layers=[3, 4, 6, 3], 
                zero_init_residual=False,groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, Pretrained=True, use_head=False, num_classes=80, use_contrast=False,
                 contrast_fea_dimension=1024, use_group=True, frozen_layers = 1):
        super(GroupModel, self).__init__()
        self.use_group = use_group
        self.num_classes = num_classes
        self.super_labels = super_labels

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.use_head = use_head
        self.use_contrast = use_contrast
        self.contrast_fea_dimension = contrast_fea_dimension
        self.inplanes = 64
        self.dilation = 1
        self.zero_init_residual = zero_init_residual
        self.frozen_layers = frozen_layers
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.group_heads = nn.ModuleList()
        self.group_classifiers = nn.ModuleList()
        self.backboneFeature = expansions[arch]

        # 保证二者必须存在一个
        assert self.use_group or self.use_head

        if self.use_group:
            # 每个Group分支拥有单独的特征提取模块 (layer 4, pooling, classifier)
            for i in range(len(super_labels)):
                self.inplanes = 2048  # self.inplanes会在self._make_layer函数中更新，为了避免这个更新带来的影响，每次循环后重新赋值
                self.group_heads.append(nn.Sequential(
                    self._make_layer(block, 512, layers[3], stride=2,
                                     dilate=replace_stride_with_dilation[2]),
                    nn.AdaptiveAvgPool2d((1, 1)),
                ))  # 其输出尺寸为  B * self.backboneFeature

                self.group_classifiers.append(
                    nn.Sequential(PFC(in_channels=self.backboneFeature, out_channels=256, dropout=0.5),
                                  ClsHead(in_channels=256, num_classes=len(super_labels[i]))
                                  ))

        if self.use_head:
            self.inplanes = 2048
            self.group_heads.append(nn.Sequential(
                self._make_layer(block, 512, layers[3], stride=2,
                                 dilate=replace_stride_with_dilation[2]),
                nn.AdaptiveAvgPool2d((1, 1)),
            ))
            self.group_classifiers.append(
                nn.Sequential(PFC(in_channels=self.backboneFeature, out_channels=256, dropout=0.5),
                              ClsHead(in_channels=256, num_classes=self.num_classes)
                              ))

        # 当使用对比学习时，增加一个特征映射模块
        self.projs = nn.ModuleList()
        if self.use_contrast:
            for i in range(len(self.super_labels)):
                self.projs.append(nn.Linear(self.backboneFeature, self.contrast_fea_dimension))

        if Pretrained:
            self.load_from_pretrain("/home/pengpeng/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth")
            print("load from pretrain weight....")
        else:
            self.init_weight()

        if self.frozen_layers > 0:
            print("frozen layers below {}".format(frozen_layers))
            self._freeze_stages()

    def _freeze_stages(self):
        self.bn1.eval()
        for m in [self.conv1, self.bn1]:
            for param in m.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_layers + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        device = (torch.device('cuda')
                  if x.is_cuda
                  else torch.device('cpu'))

        b = x.shape[0]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        feas = []
        results = []
        if self.use_head:
            temp = self.group_heads[-1](x)
            temp = torch.flatten(temp, 1)
            out1 = self.group_classifiers[-1](temp)
            results.append(out1)

        if self.use_group:
            out2 = torch.zeros(size=[b, self.num_classes]).to(device)
            for i in range(len(self.super_labels)):
                temp = self.group_heads[i](x)
                temp = torch.flatten(temp, 1)
                if self.use_contrast:
                    feas.append(self.projs[i](temp))
                out2[:, self.super_labels[i]] = self.group_classifiers[i](temp)
            results.append(out2)

        if self.use_contrast:
            feas = torch.stack(feas, dim=1)
        else:
            feas = torch.zeros(size=[20])

        return results, feas

    def load_from_pretrain(self, path):
        state_dict = torch.load(path)
        # load conv1, bn1, layer1, layer2, layer3
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

        head_state_dict = {}
        for key in unexpected_keys:
            if 'fc' not in key:
                head_state_dict['0.' + key[7:]] = state_dict[key]

        # group 分支
        if self.use_group:
            for i in range(len(self.super_labels)):
                missing_keys, unexpected_keys = self.group_heads[i].load_state_dict(head_state_dict, strict=False)

        # 主分支
        if self.use_head:
            missing_keys, unexpected_keys = self.group_heads[-1].load_state_dict(head_state_dict,strict=False)

    def freezeBackbone(self):
        for i, (name, param) in enumerate(self.named_parameters()):
            if "group" not in name:
                param.requires_grad = False

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        print("init model weight....")


def GroupModel_test():
    SUPER_GROUP = [[0, 2, 24, 26, 39, 41, 42, 43, 44, 45, 56, 57, 58, 60, 62, 63, 67, 69, 71, 72, 73, 75],
               [1, 3, 5, 7, 8, 9, 13, 15, 16, 25, 27, 28, 32, 34, 35, 38, 40, 46, 47, 48, 49, 50, 51,
                53, 55, 59, 61, 64, 65, 66, 68, 74, 77],
               [4, 6, 10, 11, 12, 14, 17, 18, 19, 20, 21, 22, 23, 29, 30, 31, 33, 36, 37, 52, 54, 70, 76, 78, 79]]

    model = GroupModel(super_labels=SUPER_GROUP, arch='resnet50', use_head=True, Pretrained=True, use_group=True)

    # print(SUPER_GROUP)
    # freezeSharedBackBone(model)
    #
    # for i, (name, param) in enumerate(model.named_parameters()):
    #     if "group" not in name:
    #         print(param.requires_grad)
    #
    # exit()

    # print(out)
    x = torch.rand(size=[3, 3, 224, 224])
    begin = time.time()
    out, feas = model(x)

    print(len(out))
    print("time",time.time() - begin)



def basemodel_test():
    model = BaseModel(arch="resnet50")
    for name, params in model.named_parameters():
        if not params.requires_grad:
            print(name)


if __name__ == '__main__':
    # basemodel_test()
    GroupModel_test()
