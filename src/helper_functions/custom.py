import collections
import os
import xml.etree.ElementTree as ET

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from src.helper_functions.sampler import get_index_cls, m2sSample, get_afterRS_pos_neg, get_base_weights

norm_transform = transforms.Compose([
    transforms.Resize(size=[224, 224]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(), ]
    # transforms.Normalize()]
)


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


VOC_CAT_MAPS = {"aeroplane": 0, "bicycle": 1, "bird": 2, "boat": 3, "bottle": 4, "bus": 5,
                "car": 6, "cat": 7, "chair": 8, "cow": 9, "diningtable": 10, "dog": 11,
                "horse": 12, "motorbike": 13, "person": 14, "pottedplant": 15, "sheep": 16,
                "sofa": 17, "train": 18, "tvmonitor": 19}


class VocMLDataset(Dataset):
    def __init__(self, root, year=2012, transform=None, target_transform=None, single_label_file=None, mask_file=None):
        self.root = root
        dataset_name = "VOC" + str(year)
        self.path_devkit = os.path.join(root, 'VOCdevkit')
        self.images_path = os.path.join(root, 'VOCdevkit', dataset_name, 'JPEGImages')
        self.annotations_path = os.path.join(root, 'VOCdevkit', dataset_name, "Annotations")
        self.transform = transform
        self.target_transform = target_transform
        self.mask_file = mask_file

        assert single_label_file is not None
        if ".txt" in single_label_file:  # VOC test dataset
            with open(single_label_file, 'r') as f:
                img_ids = f.readlines()
                self.img_ids = [img_id.strip() for img_id in img_ids]
        else:
            self.img_ids = np.load(single_label_file)

        if self.mask_file is not None:
            self.masks = np.load(mask_file)

    def __getitem__(self, idx):
        assert idx < len(self.img_ids)
        img_file_name = os.path.join(self.images_path, str(self.img_ids[idx]) + ".jpg")
        ann_file_name = os.path.join(self.annotations_path, str(self.img_ids[idx]) + '.xml')
        img = Image.open(img_file_name).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        assert os.path.exists(ann_file_name)
        assert os.path.exists(img_file_name)
        objects = self.parse_voc_xml(ET.parse(ann_file_name).getroot())['annotation']['object']

        target = np.zeros(shape=[20])
        for object in objects:
            if object['name'] not in VOC_CAT_MAPS.keys():
                print(object['name'])
                raise IndexError
            else:
                target[VOC_CAT_MAPS[object['name']]] = 1

        if self.mask_file is not None:
            return img, target, self.masks[idx]
        else:
            return img, target

    def __len__(self):
        return len(self.img_ids)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == 'annotation':
                def_dic['object'] = [def_dic['object']]
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict


if __name__ == '__main__':

    root = "/data/coco/train2017/"
    ann_file = "/data/coco/annotations/instances_train2017.json"
    LT_ann_file = "/home/pengpeng/DistributionBalancedLoss/appendix/coco/longtail2017/img_id.pkl"
    single_label_file = "/home/pengpeng/ASL/src/helper_functions/reduce_neg_new_img_ids.npy"
    mask_file = "/home/pengpeng/ASL/src/helper_functions/reduce_neg_masks.npy"
    multi_label_file = "/home/pengpeng/ASL/appendix/coco/longtail2017/img_ids.npy"

    root = "/data"

    # single_label_file = "/home/pengpeng/ASL/appendix/VOCdevkit/lt2012/single_img_ids.npy"
    # mask_file = "/home/pengpeng/ASL/appendix/VOCdevkit/lt2012/masks.npy"
    # dataset = Voc2012(root,single_label_file=single_label_file,mask_file=mask_file)

    single_label_file = "/data/VOCdevkit/VOC2007/ImageSets/Main/test.txt"
    dataset = VocMLDataset(root, year=2007, single_label_file=single_label_file)

    for i in range(len(dataset)):
        print(dataset.__getitem__(i)[1])


    exit()

    dataset = CustomDataset(root, ann_file, multi_label_file, need_masked=False, mask_file=None,
                            transform=transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(brightness=0.3,
                                                       contrast=0.3,
                                                       saturation=0.3,
                                                       hue=0),
                                # CutoutPIL(cutout_factor=0.5),
                                # transforms.ColorJitter(brightness=32,contrast=(0.5,1.5),saturation=(0.5,1.5),hue=18),
                                # RandAugment(),
                                transforms.ToTensor(),
                            ]),
                            )

    print("dataset 的数量:", len(dataset))

    base_weight = get_base_weights(dataset, 80)
    sampler = m2sSample(dataset, reduction=16, num_classes=80, weights=base_weight)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False,
        sampler=sampler,
        num_workers=8, pin_memory=True)

    pos = torch.zeros(size=[80])
    neg = torch.zeros(size=[80])
    for i, (inputData, target) in enumerate(train_loader):
        pos += torch.sum(target, dim=0)
        neg += torch.sum((1 - target), dim=0)

    # for i, (inputData, target, mask) in enumerate(train_loader):
    #
    #     pos += torch.sum(target * mask,dim=0)
    #     neg += torch.sum((1 -target) * mask,dim=0)

    print("one epoch pos number:", pos)
    print("one epoch neg number:", neg)
    exit()
    weight = pos / neg * 3

    weight = weight.cpu().numpy()
    print("weight", weight)
    np.save("./m2s_weight.npy", weight)

    exit()
    get_afterRS_pos_neg(dataset, 80)
    exit()
    get_index_cls(dataset)
    exit()
    sampler = m2sSample(dataset, reduction=16, num_classes=80)
    print(list(sampler))

    exit()
    import tqdm

    for i in tqdm.trange(len(dataset)):
        item = dataset[i]
        if len(item) == 3:
            try:
                assert np.sum(item[1] * item[2]) == 1
            except:
                print(item[1], item[2])
