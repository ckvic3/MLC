# from turtle import forward
# from pyrsistent import PRecord
# import torch
# import torch.nn as nn
# from models import make_res_layer, Bottleneck,BasicBlock
# from mmcv.runner import load_checkpoint
# from mmcv.cnn import constant_init, kaiming_init
# from torch.nn.modules.batchnorm import _BatchNorm

# class GroupModel(nn.Module):
#     def __init__(self, super_labels, block=Bottleneck, layers=[3, 4, 6, 3], Pretrained=True, 
#     use_global=False, num_classes=80, use_group=True,frozen_stages=-1):
#         super(GroupModel,self).__init__()
#         self.frozen_stages = frozen_stages
#         self.backbone = ResNet(depth=50,num_stages=4,
#                  out_indices=(0, 1, 2),
#                  style='pytorch',
#                  frozen_stages=frozen_stages)

#         if Pretrained:
#             self.backbone.init_weights(pretrained="torchvision://resnet50")
        
#         self.groupModule = GroupModule(super_labels,num_classes=num_classes,use_global=use_global)
#         self.groupModule.init_weights()

#     def forward(self,x):
#         if self.frozen_stages == 3:
#             with torch.no_grad():
#                 x = self.backbone(x)[-1]
#         else:
#             x =  self.backbone(x)[-1]
#         out = self.groupModule(x)
#         return out

# class GroupModule(nn.Module):
#     def __init__(self, super_labels,use_global,block=Bottleneck, layers=[3, 4, 6, 3], num_classes=80, use_group=True):
#         super(GroupModule, self).__init__()
#         self.use_group = use_group
#         self.num_classes = num_classes
#         self.super_labels = super_labels
#         self.use_global = use_global

#         # 保证二者必须存在一个
#         assert self.use_group or self.use_global

#         self.group_heads = nn.ModuleList()
#         self.group_classifiers = nn.ModuleList()
#         stage_with_gen_attention=((), (), (), ())
#         if self.use_group:
#             inplanes = 1024
#             planes = 512
#             num_blocks = 3
#             for i in range(len(super_labels)):
#                 res_layer = make_res_layer(
#                             block = Bottleneck,
#                             inplanes= inplanes,
#                             planes= planes,
#                             blocks= num_blocks,
#                             stride=2,
#                             dilation=1,
#                             style="pytorch",
#                             with_cp=False,
#                             conv_cfg=None,
#                             norm_cfg=dict(type='BN', requires_grad=True),
#                             gen_attention=None,
#                             gen_attention_blocks=stage_with_gen_attention[0])
                
#                 self.group_heads.append(
#                         nn.ModuleDict({"layer4":res_layer,
#                         "pool":nn.AdaptiveAvgPool2d((1,1))}
#                 ))
#                 self.group_classifiers.append(
#                     nn.Sequential(
#                         PFC(in_channels=2048, out_channels=256, dropout=0.5),
#                                   ClsHead(in_channels=256, num_classes=len(super_labels[i]))
#                                   ))

#         if self.use_global:
#             inplanes = 1024
#             planes = 512
#             res_layer = make_res_layer(
#                             Bottleneck,
#                             inplanes,
#                             planes,
#                             num_blocks,
#                             stride=2,
#                             dilation=1,
#                             style="pytorch",
#                             with_cp=False,
#                             conv_cfg=None,
#                             norm_cfg=dict(type='BN', requires_grad=True),
#                             gen_attention=None,
#                             gen_attention_blocks=())
#             self.group_heads.append(
#                         nn.ModuleDict({
#                         "layer4":res_layer,
#                         "pool":nn.AdaptiveAvgPool2d((1, 1))}
#                 ))
#             self.group_classifiers.append(
#                 nn.Sequential(
#                             PFC(in_channels=2048, out_channels=256, dropout=0.5),
#                                ClsHead(in_channels=256, num_classes=num_classes)
#                                ))


#     def init_weights(self, pretrained="torchvision://resnet50"):
#         if isinstance(pretrained, str):
#             for i in range(len(self.group_heads)):
#                 load_checkpoint(self.group_heads[i], pretrained, strict=False, logger=None)
#                 # for m in self.group_heads[i].modules():
#                 #     if isinstance(m, nn.Conv2d):
#                 #         kaiming_init(m)
#                 #     elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
#                 #         constant_init(m, 1)
#                 self.group_classifiers[i][0].init_weights()
#                 self.group_classifiers[i][1].init_weights()

#         elif pretrained is None:
#             for m in self.modules():
#                 print(m._get_name)
#                 if isinstance(m, nn.Conv2d):
#                     kaiming_init(m)
#                 elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
#                     constant_init(m, 1)

#             if self.zero_init_residual:
#                 for m in self.modules():
#                     print(m._get_name)
#                     if isinstance(m, Bottleneck):
#                         constant_init(m.norm3, 0)
#                     elif isinstance(m, BasicBlock):
#                         constant_init(m.norm2, 0)
#         else:
#             raise TypeError('pretrained must be a str or None')   


#     def forward(self,x):
#         device = (torch.device('cuda')
#                   if x.is_cuda
#                   else torch.device('cpu'))

#         b = x.shape[0]
#         results = []
#         if self.use_global:
#             temp = self.group_heads[-1]["layer4"](x)
#             temp = self.group_heads[-1]["pool"](temp)
#             temp = torch.flatten(temp, 1)
#             out1 = self.group_classifiers[-1](temp)
#             results.append(out1)

#         if self.use_group:
#             out2 = torch.zeros(size=[b, self.num_classes]).to(device)
#             for i in range(len(self.super_labels)):
#                 temp = self.group_heads[i]["layer4"](x)
#                 temp = self.group_heads[i]["pool"](temp)
#                 temp = torch.flatten(temp, 1)
#                 out2[:, self.super_labels[i]] = self.group_classifiers[i](temp)
#             results.append(out2)

#         return results,None


import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from models import PFC, ClsHead
from copy import deepcopy

def make_resnet_Layers(layers = [3,4]):
    """
    DEEP COPY SOME LAYERS OF RESNET 
    """
    base_model = resnet50(pretrained=True)
    model = nn.Sequential()
    for layer in layers:
        model.add_module(name="layer"+str(layer), module=deepcopy(getattr(base_model,"layer"+str(layer))))    
    del base_model
    return model

class GroupModel(nn.Module):
    def __init__(self,use_group=False,use_global=True,groups=[],num_classes=80,frozen_layers=-1,layers=[4]):
        """
        params:
        use_group:      使用分组分支
        use_global:     使用全局分支
        groups:         指定的分组
        num_classes:    
        fronzen_layers: backbone被固化的层数
        layers:         指定分组分支中使用的resnet layers, backbone 会自动修正其层数
        
        """
        super(GroupModel, self).__init__()
        self.num_classes = num_classes

        self.resnet = resnet50(pretrained=True)
        self.backboneFeature = 2048
        self.backbone = IntermediateLayerGetter(self.resnet,return_layers={"layer"+str(min(layers)-1):"feat"})
        
        self.frozen_layers = frozen_layers
        if self.frozen_layers > 0:
            print("fronzen backbone layer below ", self.frozen_layers)
            self._freeze_stages()

        self.use_group = use_group
        self.use_global = use_global
        self.groups = groups
        self.group_heads = nn.ModuleList()

        # 每个Group分支拥有单独的特征提取模块
        if self.use_group:
            for group in groups:
                self.group_heads.append(
                nn.Sequential(
                make_resnet_Layers(layers),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                PFC(in_channels=self.backboneFeature, out_channels=256, dropout=0.5),
                ClsHead(in_channels=256, num_classes=len(group)))
                )  

        if self.use_global:
            self.group_heads.append(
                nn.Sequential(
                make_resnet_Layers(layers),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                PFC(in_channels=self.backboneFeature, out_channels=256, dropout=0.5),
                ClsHead(in_channels=256, num_classes=self.num_classes))
            )

    def _freeze_stages(self):
        self.resnet.bn1.eval()
        for m in [self.resnet.conv1, self.resnet.bn1]:
            for param in m.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_layers + 1):
            m = getattr(self.resnet, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
    
    def forward(self,x):
        device = (torch.device('cuda')
                  if x.is_cuda
                  else torch.device('cpu'))
        b = x.shape[0]
        x = self.backbone(x)["feat"]
        
        results = []
        if self.use_global:
            global_out = self.group_heads[-1](x)
            results.append(global_out)

        if self.use_group:
            group_out = torch.zeros(size=[b, self.num_classes]).to(device)
            for i,group in enumerate(self.groups):
                group_out[:,group] = self.group_heads[i](x)
            results.append(group_out)  

        return results
