import os.path
from pycocotools.coco import COCO
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
import json
import tqdm


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


class m2sDataset(BaseDataset):
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
        super(m2sDataset, self).__init__(root, ann_file, meta_path, transform, target_transform)
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

    dataset = m2sDataset(root, annFile, meta_path, m2s_path=m2s_path, mask_path=mask_path)

    for i in tqdm.trange(len(dataset)):
        img, label, mask = dataset.__getitem__(i)

        print(type(label))
        print(type(mask))


if __name__ == '__main__':
    print("false")
