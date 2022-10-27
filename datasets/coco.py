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
import os.path
from torch.utils.data import Dataset
import json
import tqdm



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
        target = torch.max(output,dim=0)[0].float()

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
        target = torch.max(output,dim=0)[0].float()
        
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




class BaseDataset(Dataset):
    def __init__(self, root, ann_file, meta_path=None, transform=None, target_transform=None):
        super(BaseDataset, self).__init__()
        try:
            assert meta_path is not None
        except AssertionError:
            print("In Dataset, meta_path can't be None!")

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.meta_data = {}
        self.cat2cat = {}

        if os.path.exists(meta_path):
            with open(meta_path, 'r') as file:
                self.meta_data = json.loads(file.read())
            self.img_ids = list(self.meta_data.keys())
        else:
            self.coco = COCO(annotation_file=ann_file)
            # 建立 coco 类别编号到训练类别的映射
            for cat in self.coco.cats.keys():
                self.cat2cat[cat] = len(self.cat2cat)

            self.img_ids = list(self.coco.imgToAnns.keys())
            for img_id in tqdm.tqdm(self.img_ids):
                item = {}
                labels = np.zeros(shape=[80], dtype=float)
                # 逐个读取与img_id 关联的Anns，获取信息
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anns = self.coco.loadAnns(ann_ids)
                for ann in anns:
                    labels[self.cat2cat[ann['category_id']]] = 1

                # json.dump 不能处理 np.array 对象
                item['gt_label'] = labels.tolist()
                item['file_name'] = self.coco.loadImgs(img_id)[0]['file_name']

                self.meta_data[str(img_id)] = item

            with open(file=meta_path, mode='w') as f:
                f.write(json.dumps(self.meta_data))

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = self._get_img(img_id)
        labels = self._get_label(img_id)

        return img, labels

    def _get_label(self, img_id) -> object:
        labels = torch.tensor(self.meta_data[img_id]['gt_label'])
        if self.target_transform is not None:
            labels = self.target_transform(labels)
        return labels

    def _get_img(self, img_id) -> object:
        path = self.meta_data[img_id]['file_name']
        path = os.path.join(self.root, path)
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_ids)


class m2sCoco(BaseDataset):
    def __init__(self, root, ann_file, meta_path=None, transform=None, target_transform=None, m2s_path=None,
                 mask_path=None):
        """
        Args:
            root:       image root path
            ann_file:   ann file path
            meta_path:  meta file path, meta_data  { img_id: {gt_label , file_name}}
            transform:
            target_transform:
            m2s_path:   multi to single label , .npy file
            mask_path:  masks used in multi to single label, .npy file
        """
        super(m2sCoco, self).__init__(root, ann_file, meta_path, transform, target_transform)
        try:
            assert m2s_path is not None
            assert mask_path is not None
        except AssertionError:
            print("m2s and mask path can't be None!")

        self.img_ids = np.load(m2s_path)
        self.masks = np.load(mask_path)

    def __getitem__(self, idx):
        img_id = str(self.img_ids[idx])
        img = self._get_img(img_id)
        label = self._get_label(img_id)
        mask = self.masks[idx]
        return img, label, mask

    def __len__(self):
        return len(self.img_ids)


def test_BaseDataset():
    root = "/data/coco/train2014"
    annFile = os.path.join("/data/coco/", 'annotations2014/annotations/instances_train2014.json')
    meta_path = "/home/pengpeng/ASL/appendix/coco/coco_train_2014.json"
    dataset = BaseDataset(root, annFile, meta_path)

    for i in tqdm.trange(len(dataset)):
        img, label = dataset.__getitem__(i)


def test_m2sDataset():
    root = "/data/coco/train2017"
    annFile = os.path.join("/data/coco/", 'annotations/instances_train2017.json')
    meta_path = "/home/pengpeng/ASL/appendix/coco/coco_train_2017.json"
    m2s_path = "/home/pengpeng/ASL/src/helper_functions/new_img_ids.npy"
    mask_path = "/home/pengpeng/ASL/src/helper_functions/masks.npy"

    dataset = m2sCoco(root, annFile, meta_path, m2s_path=m2s_path, mask_path=mask_path)

    for i in tqdm.trange(len(dataset)):
        img, label, mask = dataset.__getitem__(i)

        mask = torch.tensor(mask)
        print((label*mask).sum())






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
