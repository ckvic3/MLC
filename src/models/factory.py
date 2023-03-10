from .group import BaseModel,CBAMModel
from models import GroupModel
from mmcv import Config
from .decoder import DecoderModel

def createModelFactory(cfg:Config):
    print("using {} model".format(cfg.model['name']))
    if cfg.model['name'] == "base":
        return BaseModel(**cfg.model['param'])
    elif cfg.model['name'] == "group":
        return GroupModel(**cfg.model['param'])
    elif cfg.model['name'] == "cbam":
        return CBAMModel(**cfg.model['param'])
    elif cfg.model['name'] =='decoder':
        return DecoderModel(**cfg.model['param'])