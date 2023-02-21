import torchvision.transforms as transforms
import os
from .voc import VocDataset, copyDecouplingVocDataset 
from .coco import copyDecouplingCOCODataset,LTCOCODataset,CocoDataset
from typing import Tuple
import torch
from .sampler import ClassAwareSampler
from randaugment import RandAugment
from src.helper_functions.helper_functions import CutoutPIL

def getDataloader(cfg):
    useCopyDecoupling = cfg.dataset['useCopyDecoupling']
    imageSize = cfg.dataset['imageSize']
    root = cfg.dataset['root']

    if cfg.dataset['name'] == "coco":
        trainDataset,valDataset = getCOCODataset(useCopyDecoupling,imageSize,root)
    elif cfg.dataset['name'] == 'voc':
        trainDataset,valDataset = getVocDataset(useCopyDecoupling,imageSize,root)
    else:
        raise NotImplementedError
    
    if cfg.dataset['sampler'] is not None:
        print("using ClassAware Sampler")
        sampler = ClassAwareSampler(data_source=trainDataset,clsDataListFile=cfg.dataset['clsDataListFile'])
        shuffle = False
    else:
        sampler = None 
        shuffle = True

    train_loader = torch.utils.data.DataLoader(
        trainDataset, batch_size=cfg.batch_size, shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
        valDataset, batch_size=cfg.batch_size * 4, shuffle=False,
        num_workers=cfg.workers, pin_memory=False)
    
    return train_loader, val_loader

def getCOCODataset(useCopyDecoupling=False, imageSize=224,root=None):
    dataPathTrain = f'{root}/train2017'
    dataPathVal = f'{root}/val2017'

    instancesPathTrain = os.path.join(
        root, 'annotations/instances_train2017.json')
    instancesPathVal = os.path.join(
        root, 'annotations/instances_val2017.json')

    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    trainTransform = transforms.Compose([
                                        transforms.Resize((imageSize, imageSize)),
                                        # transforms.RandomHorizontalFlip(),
                                        CutoutPIL(cutout_factor=0.5),
                                        RandAugment(),
                                        transforms.ToTensor(),
                                        normalization
                                    ])
    testTransform = transforms.Compose([
                                    transforms.Resize((imageSize, imageSize)),
                                    transforms.ToTensor(),
                                    normalization
                                ])
    if useCopyDecoupling:
        imgIdFile = "./appendix/coco/longtail2017/copyDecoupling/imgIds.txt"
        maskFile = "./appendix/coco/longtail2017/copyDecoupling/masks.npy"
        trainDataset = copyDecouplingCOCODataset(dataPathTrain, instancesPathTrain,
            transform=trainTransform,
            imgIdFile=imgIdFile, maskFile=maskFile)
    else:
        trainDataset = LTCOCODataset(dataPathTrain, instancesPathTrain,
                           imgIdFile="./appendix/coco/longtail2017/img_id.txt",
                           transform=trainTransform)

    valDataset = CocoDataset(dataPathVal,
                                instancesPathVal,
                                testTransform
                            )
    
    print("len(trainDataset)): ", len(trainDataset))
    print("len(valDataset)): ", len(valDataset))
    return trainDataset, valDataset



def getVocDataset(useCopyDecoupling=False,imageSize:int=224,root=None) ->Tuple[VocDataset, VocDataset]:
    root = os.path.join(root,"/VOCdevkit/VOC2012/JPEGImages")
    annoFile = os.path.join(root,"VOCdevkit/VOC2012/Annotations/")
    imgIdFile = "./appendix/VOCdevkit/longtail2012/img_id.txt"

    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    trainTransform = transforms.Compose([
                                        transforms.Resize((imageSize, imageSize)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalization
                                    ])
    testTransform = transforms.Compose([transforms.Resize((imageSize, imageSize)),
                                    transforms.ToTensor(),
                                    normalization])

    if useCopyDecoupling:
        imgIdFile = "./appendix/VOCdevkit/longtail2012/copyDecoupling/img_id.txt"
        maskFile = "./appendix/VOCdevkit/longtail2012/copyDecoupling/masks.npy"
        trainDataset = copyDecouplingVocDataset(root,annoFile,imgIdFile,maskFile,transform=trainTransform)
    else:
        trainDataset = VocDataset(root,annoFile,imgIdFile,transform=trainTransform)
    
    testRoot = os.path.join(root,"VOCdevkit/VOC2007/JPEGImages") 
    testAnnoFile = os.path.join(root,"VOCdevkit/VOC2007/Annotations/")
    testImgIdFile = "./appendix/VOCdevkit/longtail2012/test.txt"
    testDataset = VocDataset(testRoot,testAnnoFile,testImgIdFile,transform=testTransform)
    
    print("len(train_dataset)): ", len(trainDataset))
    print("len(val_dataset)): ", len(testDataset))
    return trainDataset, testDataset


