import os.path

import mmcv
from pycocotools.coco import COCO
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
import json
import tqdm


class CustomDataset(Dataset):
    def __init__(self, root, ann_file, single_label_file, transform=None, target_transform=None,
                 need_masked=False, mask_file=None):
        self.root = root
        self.coco = COCO(annotation_file=ann_file)
        self.need_masked = need_masked
        self.mask_file = mask_file
        self.transform = transform
        self.target_transform = target_transform

        # 构建COCO类别id映射
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i
            for i, cat_id in enumerate(self.cat_ids)
        }

        if self.need_masked:
            assert self.mask_file is not None
            # key img_id, value mask vector
            self.masks = np.load(self.mask_file)

        if single_label_file is not None:
            self.img_ids = np.load(single_label_file)
        else:
            self.img_ids = self.coco.imgToAnns.keys()

    def __getitem__(self, idx):
        img_id = int(self.img_ids[idx])
        # 读取图片
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.root, img_path)
        img = Image.open(img_path).convert('RGB')

        # 获取标注信息
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        ann_infos = self.coco.loadAnns(ann_ids)

        target = torch.zeros(size=[80])
        for ann_info in ann_infos:
            target[self.cat2label[ann_info['category_id']]] = 1

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target = np.array(target)

        # 需要读取mask
        if self.need_masked:
            mask = self.masks[idx]
            return img, target, mask
        else:
            return img, target

    def __len__(self):
        return len(self.img_ids)

class SDataset(Dataset):
    def __init__(self, root, ann_file, img_ids, gt_labels, transform=None):
        """
        适用于已经提前获取了要使用的img_ids 和 gt_labels

        :param root: 图片的根目录i
        :param ann_file:
        :param img_ids: 需要使用的img id
        :param transform:
        :param gt_labels: numpy 形式的 真实标签，需要和img_ids 中的下标一一对应
        """
        super(SDataset, self).__init__()
        self.root = root
        self.coco = COCO(annotation_file=ann_file)
        self.transform = transform
        self.img_ids = img_ids
        self.gt_labels = gt_labels

    def __getitem__(self, idx):
        img_id = int(self.img_ids[idx])
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.root, img_path)
        img = Image.open(img_path).convert('RGB')

        target = self.gt_labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.img_ids)


class FlexibleDataset(Dataset):
    def __init__(self, root, ann_file, img_ids, transform=None,
                 need_masked=False, mask_file=None):
        """
        适用于已经提前获取了要使用的img_ids

        :param root: 图片的根目录i
        :param ann_file:
        :param img_ids: 需要使用的img id
        :param transform:
        :param need_masked:
        :param mask_file:
        """
        self.root = root
        self.coco = COCO(annotation_file=ann_file)
        self.transform = transform

        # 构建COCO类别id映射
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i
            for i, cat_id in enumerate(self.cat_ids)
        }

        self.img_ids = img_ids

    def __getitem__(self, idx):
        img_id = int(self.img_ids[idx])

        # 读取图片
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.root, img_path)
        img = Image.open(img_path).convert('RGB')

        # 获取标注信息
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        ann_infos = self.coco.loadAnns(ann_ids)

        target = torch.zeros(size=[80])
        for ann_info in ann_infos:
            target[self.cat2label[ann_info['category_id']]] = 1

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.img_ids)


def SDataset_test():
    metadata = mmcv.load("../appendix/coco/longtail2017/metadata.pkl")

    root = "/data/coco/"

    data_path_train = f'{root}/train2017'
    instances_path_train = os.path.join(root, 'annotations/instances_train2017.json')

    train_dataset = SDataset(data_path_train,
                             instances_path_train,
                             metadata["img_ids"],
                             metadata["gt_labels"],
                             None)

    print(train_dataset[1])


def FlexibleDataset_test():
    metadata = mmcv.load("../appendix/coco/longtail2017/metadata.pkl")
    root = "/data/coco/"
    data_path_train = f'{root}/train2017'
    instances_path_train = os.path.join(root, 'annotations/instances_train2017.json')

    train_dataset = FlexibleDataset(data_path_train,
                                    instances_path_train,
                                    metadata["img_ids"],
                                    None)
    print(train_dataset[1])


if __name__ == '__main__':
    FlexibleDataset_test()
