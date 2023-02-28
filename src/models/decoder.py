import torch
import torch.nn as nn
from .resnet import resnet50
from ..ml_decoder.ml_decoder import MLDecoder

class DecoderModel(resnet50):
    def __init__(self, num_classes=80, pretrained=True,num_of_groups=-1):
        super().__init__(pretrained=pretrained,num_classes=80)
        self.avgpool = nn.Identity()
        if num_of_groups != -1:
            self.fc = MLDecoder(num_classes=80, num_of_groups=num_of_groups,initial_num_features=2048, decoder_embedding=768, zsl=False)    
        else:
            self.fc = MLDecoder(num_classes=80, num_of_groups=num_classes,initial_num_features=2048, decoder_embedding=768, zsl=False)    
            