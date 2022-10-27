import random

import mmcv
import torch
from torch.utils.data.sampler import Sampler
import numpy as np
from torch.utils.data import Sampler, WeightedRandomSampler
import os
from tqdm import trange

class RandomCycleIter:
    def __init__(self, data_list, test_mode=False):
        self.data_list = list(data_list)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1

        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)

        return self.data_list[self.i]


def class_aware_sample_generator(cls_iter, data_iter_list, n, num_samples_cls=1):
    """

    :param cls_iter: 类别迭代器
    :param data_iter_list:  每个类别数据的迭代器
    :param n: 采样得到的总样本数
    :param num_samples_cls: 每一次连续得到的样本数 为了提高采样效率
    :return:
    """

    i = 0
    j = 0
    while i < n:

        #         yield next(data_iter_list[next(cls_iter)])

        if j >= num_samples_cls:
            j = 0

        if j == 0:
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]] * num_samples_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]

        i += 1
        j += 1


# Distributed Balanced loss Class aware Sampler
class ClassAwareSampler(Sampler):
    """
    meta_data:  cls_data_list 需要包含每个类别的所包含的样本idx, 
                gt_labels 
    """
    def __init__(self, data_source, meta_data=None, reduction=4, num_samples_cls=3):
        random.seed(0)
        torch.manual_seed(0)
        self.data_source = data_source
        self.meta_data = meta_data
        self.reduction = reduction
        # 每次连续采样样本数
        self.num_samples_cls = num_samples_cls
        self.epoch = 0
        
        self.cls_data_list = self.meta_data["cls_data_list"]
        
        # 画图需要
        self.gt_labels = self.meta_data["gt_labels"]

        self.num_classes = len(self.cls_data_list)

        # 类别选择的random Sampler,输出随机的返回值
        self.class_iter = RandomCycleIter(list(range(self.num_classes)))

        # 从每个类数据选择的random Sampler
        self.data_iter_list = [RandomCycleIter(x) for x in self.cls_data_list]

        self.num_samples = len(self.gt_labels) * 10 
        # self.num_samples = int(max([len(x) for x in self.cls_data_list])) * self.num_classes // self.reduction
        

    def __iter__(self):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list, self.num_samples,
                                            self.num_samples_cls)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_samples

    # 画图的辅助函数
    def get_sample_per_class(self):
        condition_prob = np.zeros([self.num_classes, self.num_classes])
        sample_per_cls = np.asarray([len(x) for x in self.gt_labels])
        rank_idx = np.argsort(-sample_per_cls)

        for i, cls_labels in enumerate(self.gt_labels):
            num = len(cls_labels)
            condition_prob[i] = np.sum(np.asarray(cls_labels), axis=0) / num

        sum_prob = np.sum(condition_prob, axis=0)
        need_sample = sample_per_cls / sum_prob
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.bar(range(self.num_classes), sum_prob[rank_idx], alpha=0.5, color='green', label='sum_j( p(i|j) )')
        plt.legend()
        plt.hlines(1, 0, self.num_classes, linestyles='dashed', color='r', linewidth=1)
        ax2 = fig.add_subplot(2, 1, 2)
        # ax2.bar(range(self.num_classes), need_sample[rank_idx], alpha = 0.5, label='need_avg')
        ax2.bar(range(self.num_classes), sample_per_cls[rank_idx], alpha=0.5, label='ori_distribution')
        plt.legend()
        plt.savefig('./coco_resample_deduce.jpg')
        print('saved at ./coco_resample_deduce.jpg')
        print(np.min(sum_prob), np.max(need_sample))
        exit()


def sampler_test():
    # from dataset import SDataset
    import os
    from datasets.coco import LTCOCO
    import mmcv
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms

    metadata = mmcv.load("../appendix/coco/longtail2017/metadata.pkl")
    root = "/data/coco/"

    data_path_train = f'{root}/train2017'
    instances_path_train = os.path.join(root, 'annotations/instances_train2017.json')

    # train_dataset = SDataset(data_path_train,
    #                          instances_path_train,
    #                          metadata["img_ids"],
    #                          metadata["gt_labels"],
    #                          transform=transforms.Compose([
    #                              transforms.Resize((224, 224)),
    #                              transforms.ToTensor(),
    #                          ]))

    train_dataset = LTCOCO(root="/home/share1/coco/train2017",
                           annFile="/home/share1/coco/annotations/instances_train2017.json",
                           LT_ann_file="/home/pengpeng/MLC/appendix/coco/longtail2017/img_id.pkl",
                           transform=transforms.Compose([
                               transforms.Resize((224, 224)),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                           ])
                           )

    sampler = ClassAwareSampler(train_dataset, metadata, reduction=20)
    class_aware_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, sampler=sampler, num_workers=2)

    results = torch.zeros(size=[80])

    for i, (imgs, targets) in enumerate(class_aware_loader):
        results += torch.sum(targets.max(dim=1)[0], dim=0)
    #    这里显示的是比率值，与最大值的比率
    print(results)
    print("class aware loader total", results.sum())
    print("class aware loader", results / torch.max(results))

    loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

    results = torch.zeros(size=[80])
    for i, (imgs, targets) in enumerate(loader):
        results += torch.sum(targets.max(dim=1)[0], dim=0)
    print(results)
    print("base loader total", results.sum())
    print("base loader", results / torch.max(results))

    

class m2sSample(WeightedRandomSampler):
    def __init__(self,data_source,reduction,num_classes,weights=None,
                 weights_save_path="/home/pengpeng/ASL/appendix/coco/lt2017/sample_weights.npy",
                 class_freq_file="/home/pengpeng/ASL/appendix/coco/longtail2017/class_freq.pkl"):
        # 默认设置的为 COCO-MLT
        self.num_samples = int(len(data_source) * num_classes / reduction)
        if weights is None:
            weights = get_weights(data_source,num_classes,weights_save_path,class_freq_file).tolist()
        else:
            weights = weights.tolist()
        super(m2sSample, self).__init__(weights,self.num_samples,replacement=True)

    def __len__(self):
        return self.num_samples


def get_weights(dataset, num_classes, weights_save_path="/home/pengpeng/ASL/appendix/coco/lt2017/sample_weights.npy",
                class_freq_file="/home/pengpeng/ASL/appendix/coco/longtail2017/class_freq.pkl"):
    """
        获取 mutli to single 数据集的 class balance 采样时每个样本的采样概率。
        Args:
            dataset: Custom dateset
            num_classes:

        Returns:
            sample_weights
        """
    if os.path.exists(weights_save_path):
        sample_weigths = np.load(weights_save_path)
    else:
        class_freq = mmcv.load(class_freq_file)['class_freq']
        class_freq = np.array(class_freq)
        weights = np.sum(class_freq) / class_freq
        sample_weigths = []
        for i in trange(len(dataset)):
            img, target, mask = dataset[i]
            target = target * mask
            if target.sum()>1:
                print("idx",i,"target",target,"mask",mask)
            for j in range(num_classes):
                if target[j] == 1:
                    sample_weigths.append(weights[j])
        sample_weigths = np.array(sample_weigths)
        np.save(weights_save_path, sample_weigths)
    print(sample_weigths.shape)
    return sample_weigths

if __name__ == '__main__':
    metadata = mmcv.load("../appendix/coco/longtail2017/metadata.pkl")
    gt_labels = np.array(metadata["gt_labels"])
    numbers = np.sum(gt_labels, axis=0)
    print(metadata.keys())
    cls_data_list = metadata["cls_data_list"]
    # assert 全部成立
    for i in range(numbers.shape[0]):
        assert len(metadata["cls_data_list"][i]) == numbers[i]
    sampler_test()

"""
使用class aware sampler 后每个类别的样本数量
[2868.,  138.,  796.,   80.,   73.,  130.,   70.,  283.,   68.,  138.,
  73.,   73.,   62.,  253.,   91.,  111.,  168.,   70.,   62.,   65.,
  60.,   56.,   59.,   64.,  334.,  127.,  582.,   95.,   95.,   57.,
  81.,   72.,  129.,   64.,   98.,  102.,   58.,   62.,   70., 1006.,
 182., 1061.,  267.,  429.,  381.,  760.,  126.,  125.,  126.,  125.,
  86.,   95.,   72.,   93.,   68.,  145., 1230.,  304.,  393.,   97.,
1297.,   72.,  422.,  262.,  170.,  170.,  191.,  228.,  231.,  352.,
  71.,  500.,  252.,  594.,  221.,  204.,   72.,   84.,   79.,   80.]

"""
