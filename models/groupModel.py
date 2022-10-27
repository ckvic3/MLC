import torch
import torch.nn as nn

from models import 
class GroupModule(nn.Module):
    def __init__(self, super_labels, block=Bottleneck, layers=[3, 4, 6, 3], Pretrained=True, 
    use_global=False, num_classes=80, use_group=True):
        super(GroupModel, self).__init__()
        self.use_group = use_group
        self.num_classes = num_classes
        self.super_labels = super_labels
        norm_layer = nn.BatchNorm2d
        self.use_global = use_global

        replace_stride_with_dilation = False
        # 保证二者必须存在一个
        assert self.use_group or self.use_global

        self.groups_heads.append(nn.sequential
            )
        if self.use_global:
            for i in range(len(super_labels)):
                res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                gen_attention=gen_attention,
                gen_attention_blocks=stage_with_gen_attention[i])
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

    def forward(self,x):
        


        
