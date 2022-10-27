import json
import os
from copy import deepcopy
import random
import time
from copy import deepcopy
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.datasets as dset
from torchvision import datasets as datasets
import torch
from PIL import ImageDraw
from pycocotools.coco import COCO
import torch.utils.data as data
from collections import OrderedDict


def seed_torch(seed):
    os.environ['PYTHONASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def MixUPForML(imageA, imageB, labelA, labelB, alpha=0.5):
    """
    MixUp for multi label images
    Args:
        imageA:
        imageB:
        labelA:
        labelB:
        alpha:

    Returns:
        image, label
    """
    image = alpha * imageA + (1-alpha) * imageB
    label = alpha * labelA + (1-alpha) * labelB
    return image, label


class MixUPDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, use_mixup=False, alpha=0.5):
        self.dataset = dataset
        self._use_mixup = use_mixup
        self._alpha = alpha

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label, mask = self.dataset[idx]
        if self._use_mixup:
            idx2 = np.random.choice(np.delete(np.arange(len(self)),idx))
            image2, label2, _ = self.dataset[idx2]
            image, label = MixUPForML(image,image2,label,label2,self._alpha)

        return image,label, mask


def parse_args(parser):
    # parsing args
    args = parser.parse_args()
    if args.dataset_type == 'OpenImages':
        args.do_bottleneck_head = True
        if args.th == None:
            args.th = 0.995
    else:
        args.do_bottleneck_head = False
        if args.th == None:
            args.th = 0.7
    return args


def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean(), ap


class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.sum = None
        self.cnt = None
        self.avg = None
        self.ema = None
        self.initialized = False

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def initialize(self, val, n):
        self.val = val
        self.sum = val * n
        self.cnt = n
        self.avg = val
        self.ema = val
        self.initialized = True

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.ema = self.ema * 0.99 + self.val * 0.01


class CocoDetection(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)
        # print("img keys number ", len(list(sorted(self.coco.imgs.keys())))) 5000
        # self.ids = list(self.coco.imgToAnns.keys())
        self.ids = list(self.coco.imgs.keys())
        # print("imgToAnns number ", self.__len__()) 4952
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


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


# most borrow from: https://github.com/HCPLab-SYSU/SSGRL/blob/master/utils/metrics.py

import numpy as np


def voc_ap(rec, prec, true_num):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # print("mrec", mrec[i])
    # print("mpre", mpre[i])

    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_mAP(imagessetfilelist, num, return_each=False):
    """
    if return_each: return mAP, aps
    else:   return mAP
    """
    if isinstance(imagessetfilelist, str):
        imagessetfilelist = [imagessetfilelist]
    lines = []
    for imagessetfile in imagessetfilelist:
        with open(imagessetfile, 'r') as f:
            lines.extend(f.readlines())

    seg = np.array([x.strip().split(' ') for x in lines]).astype(float)
    gt_label = seg[:, num:].astype(np.int32)
    num_target = np.sum(gt_label, axis=1, keepdims=True)

    sample_num = len(gt_label)
    class_num = num
    tp = np.zeros(sample_num)
    fp = np.zeros(sample_num)
    aps = []

    for class_id in range(class_num):
        confidence = seg[:, class_id]
        sorted_ind = np.argsort(-confidence)  # 按照confidence从大到小排序
        sorted_scores = np.sort(-confidence)
        sorted_label = [gt_label[x][class_id] for x in sorted_ind]  # 得到对应的真实label

        for i in range(sample_num):
            tp[i] = (sorted_label[i] > 0)
            fp[i] = (sorted_label[i] <= 0)
        true_num = 0
        true_num = max(sum(tp), 1)
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(true_num)  # 按照排序 计算召回率
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)  # 按照排序 计算精度
        ap = voc_ap(rec, prec, true_num)
        aps += [ap]

    np.set_printoptions(precision=6, suppress=True)
    aps = np.array(aps) * 100
    mAP = np.mean(aps)
    if return_each:
        return mAP, aps
    return mAP


def str2list(str_cats):
    rets = str.split(str_cats, ',')
    l = []
    for i in range(len(rets)):
        l.append(int(rets[i]))
    return l


class CocoBalancedDetection(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.coco = CocoDetection(root, annFile)
        try:
            self.indexes = np.load('/home/pengpeng/ASL/my_sta/balanced.npy')
            self.masks = np.load('/home/pengpeng/ASL/my_sta/balanced_masks.npy')
            self.labelSet = np.load('/home/pengpeng/ASL/my_sta/balanced_labels.npy')
        except:
            print("read data error!")
            exit()
        self.input_transform = transform

    def __getitem__(self, index):
        image = self.coco[self.indexes[index]][0]
        target = self.labelSet[index]
        mask = self.masks[index]
        if self.input_transform is not None:
            image = self.input_transform(image)
        return image, torch.from_numpy(target), torch.from_numpy(mask)

    def __len__(self):
        return self.indexes.shape[0]


category_map = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "13": 12, "14": 13, "15": 14, "16": 15, "17": 16, "18": 17, "19": 18, "20": 19, "21": 20, "22": 21, "23": 22, "24": 23, "25": 24, "27": 25, "28": 26, "31": 27, "32": 28, "33": 29, "34": 30, "35": 31, "36": 32, "37": 33, "38": 34, "39": 35, "40": 36, "41": 37, "42": 38, "43": 39, "44": 40, "46": 41, "47": 42, "48": 43, "49": 44, "50": 45, "51": 46, "52": 47, "53": 48, "54": 49, "55": 50, "56": 51, "57": 52, "58": 53, "59": 54, "60": 55, "61": 56, "62": 57, "63": 58, "64": 59, "65": 60, "67": 61, "70": 62, "72": 63, "73": 64, "74": 65, "75": 66, "76": 67, "77": 68, "78": 69, "79": 70, "80": 71, "81": 72, "82": 73, "84": 74, "85": 75, "86": 76, "87": 77, "88": 78, "89": 79, "90": 80}

class COCODataset(data.Dataset):
    def __init__(self, root, annFile, input_transform=None,
                 labels_path="./data/train_labels.npy",
                 used_category=-1):
        self.coco = CocoDetection(root, annFile)
        self.ids = self.coco.ids
        self.category_map = category_map
        self.input_transform = input_transform
        self.labels_path = labels_path
        self.used_category = used_category

        self.labels = []
        dicts = {}
        for i in range(80):
            dicts[i] = {}
        if os.path.exists(self.labels_path):
            self.labels = np.load(self.labels_path).astype(np.float64)
            self.labels = (self.labels > 0).astype(np.float64)
        else:
            print("No preprocessed label file found in {}.".format(self.labels_path))
            l = len(self.coco)
            for i in tqdm(range(l)):
                img_id = self.ids[i]
                item = self.coco[i]
                label = item[1].max(dim=0)[0].cpu().numpy()
                for j in range(80):
                    if label[j] == 1.0:
                        dicts[j][i] = img_id
                self.labels.append(label)
            self.save_datalabels(labels_path)
            os.makedirs('./coco_sta', exist_ok=True)
            import pandas as pd
            for index in tqdm(range(80)):
                data = np.array(list(dicts[index].items()))
                df = pd.DataFrame(data)
                df.to_csv('./coco_sta/{}.csv'.format(index))


    def __getitem__(self, index):
        input = self.coco[index][0]
        if self.input_transform:
            input = self.input_transform(input)
        return input, self.labels[index]

    def getCategoryList(self, item):
        categories = set()
        for t in item:
            categories.add(t['category_id'])
        return list(categories)

    def getLabelVector(self, categories):
        label = np.zeros(80)
        # label_num = len(categories)
        for c in categories:
            index = self.category_map[str(c)] - 1
            label[index] = 1.0  # / label_num
        return label

    def __len__(self):
        return len(self.coco)

    def save_datalabels(self, outpath):
        """
            Save datalabels to disk.
            For faster loading next time.
        """
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        labels = np.array(self.labels)
        np.save(outpath, labels)


def clean_state_dict(state_dict):
    """ function to modify state prefix name
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict


class COCO_Dataset(datasets.coco.CocoDetection):
    def __init__(self,root, annFile, transform=None, target_transform = None,
                 target_file_name ="/home/pengpeng/ASL/train2014Modified40.json",
                 mask_file_name ="/home/pengpeng/ASL/train2014ModifiedMask40.json"):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform

        print("load labelVectors...")
        file_name = target_file_name
        with open(file_name, 'r') as json_file:
            json_data = json_file.read()
            # self.targets is img_id : labelVector
            self.targets = json.loads(json_data)

        self.ids = list(self.targets.keys())
        file_name = mask_file_name
        with open(file_name, 'r') as json_file:
            json_data = json_file.read()
            self.masks = json.loads(json_data)


    def __getitem__(self, index):
        img_id = int(self.ids[index])
        target = np.array(self.targets[str(img_id)])
        # mask = np.array(self.masks[str(img_id)])
        mask = np.zeros(shape=[80])
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, mask.astype(dtype=int)

    def __len__(self):
        return len(self.ids)

def dataCleaning():
    train_json_path = os.path.join("/data/coco/", 'annotations2014/annotations/instances_train2014.json')

    labels_path = "/home/pengpeng/ASL/src/helper_functions/data/train_labels.npy"
    dataset = COCODataset("/data/coco/train2014", train_json_path)


# voycn.com/article/gerensikao-shenduxueximoxingwendingkefuxianshedingsuijizhongzidaquan
def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


class WarmUpLR(torch.optim.lr_scheduler._LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


if __name__ == '__main__':
    train_json_path = os.path.join("/data/coco/", 'annotations2014/annotations/instances_train2014.json')
    dataset = COCO_Dataset(root="/data/coco/train2014",annFile=train_json_path,
                           target_file_name="/home/pengpeng/ASL/train2014super_label_1.json")
    for i in tqdm(range(len(dataset))):
        img, target, _ = dataset[i]
        print(target)
        break
    # labelSet = np.load('/home/pengpeng/multi-label-classification/query2labels/my_sta/balanced_labels.npy')
    # print(labelSet.shape)
    #
    # print(labelSet[23].shape)
    # a = np.zeros([80, 1])
    # b = a.reshape([1, 80])
    # print(b)
    #
    # indexes = np.load('/home/pengpeng/multi-label-classification/query2labels/my_sta/balanced.npy')
    # print(indexes.shape)
    #
    # train_json_path = os.path.join("/data/coco/",'annotations2014/annotations/instances_train2014.json')
    # dataset = CoCoDataset(root ="/data/coco/train2014",
    #             annFile=train_json_path)
    #
    # print(len(dataset))
    # for i in range(len(dataset)):
    #     var = dataset[i]

    # the length of CoCoDataset
    # coco = COCO(train_json_path)
    # ids = list(sorted(coco.imgs.keys()))
    # another_ids = list(coco.imgToAnns.keys())
    # print(len(ids))
    # print(len(another_ids))

    # coco2014 = np.load("/home/pengpeng/multi-label-classification/query2labels/data/coco/train_label_vectors_coco14.npy")
    # print(coco2014.shape)
    # not_appear_ids = np.load("./not_appear_ids.npy")
    # print(not_appear_ids.shape)

    # not_appear_ids = []
    # for id in ids:
    #     if id not in another_ids:
    #         print(id)
    #         items = coco.getAnnIds(imgIds=id)
    #         print(items)
            # not_appear_ids.append(id)

    # not_appear_ids =np.array(not_appear_ids)
    # np.save("./not_appear_ids.npy",not_appear_ids)

