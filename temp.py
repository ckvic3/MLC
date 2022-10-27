from src.models import GroupModel_test
from models import ResNet
import mmcv
import torch
import time

backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch')


model = ResNet(depth=50,num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 gen_attention=None,
                 )

x = torch.rand(size=[3, 3, 224, 224])
begin = time.time()
out = model(x)

print(out[-1].shape)
print("time",time.time() - begin)
