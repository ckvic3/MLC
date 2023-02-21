import os
import os.path
import mmcv
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torchvision import datasets as datasets

class CocoDataset(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)

        self.ids = self.coco.getImgIds()
        # 采用下述方式得到的数量与ASL中提到的数量一致
        # self.ids = list(self.coco.imgToAnns.keys())

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
        #  target shape [80]
        target = torch.max(output,dim=0)[0].float()

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, target

    def __len__(self) -> int:
        return len(self.ids)

class LTCOCODataset(CocoDataset):
    def __init__(self, root, annFile, transform=None, target_transform=None, imgIdFile=None):
        super().__init__(root, annFile, transform, target_transform)
        self.ids = mmcv.load(imgIdFile)

class copyDecouplingCOCODataset(LTCOCODataset):
    def __init__(self, root, annFile, transform=None, target_transform=None, imgIdFile=None, maskFile=None):
        super().__init__(root, annFile, transform, target_transform, imgIdFile)
        self.masks = np.load(maskFile)

    def __getitem__(self, idx):
        img, label, _ = super().__getitem__(idx)
        mask = self.masks[idx]
        return img, label, mask

    
