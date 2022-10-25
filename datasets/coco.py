import os
from copy import deepcopy
import random
import time
from copy import deepcopy

import numpy as np
import torchvision.transforms as transforms
import mmcv
from PIL import Image
from torchvision import datasets as datasets
import torch
from PIL import ImageDraw
from pycocotools.coco import COCO


class CocoDetection(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = self.coco.getImgIds()
        # self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # print(self.cat2cat)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        target = output

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class LTCOCO(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, LT_ann_file=None, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)

        if LT_ann_file is not None:
            img_ids = []
            img_ids += mmcv.list_from_file(LT_ann_file)
            self.ids = img_ids
        else:
            self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)

    def __getitem__(self, index):
        coco = self.coco
        img_id = int(self.ids[index])

        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        target = output

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class GroupDataset(LTCOCO):
    def __init__(self, root, annFile, LT_ann_file=None, transform=None, target_transform=None, img_ids=None):
        super(GroupDataset, self).__init__(root, annFile, LT_ann_file, transform, target_transform)
        self.ids = img_ids


if __name__ == '__main__':
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    dataset = LTCOCO(root="/home/share1/coco/",
                     annFile="/home/share1/coco/annotations/instances_train2017.json",
                     LT_ann_file="/home/pengpeng/MLC/appendix/coco/longtail2017/img_id.pkl",
                     transform=transforms.Compose([
                         transforms.Resize((224, 224)),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         normalization
                     ])
                     )

    print(len(dataset))
