from torchvision.models import resnet50 as resnet
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter

class resnet50(nn.Module):
    def __init__(self, num_classes=80, pretrained=True):
        super(resnet50,self).__init__()
        self.model = IntermediateLayerGetter(resnet(pretrained=pretrained), {'layer4': 'out'})        
        self.avgpool = None
        self.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        x = x['out']
        x = self.avgpool(x)
        x = self.fc(x)
        return x