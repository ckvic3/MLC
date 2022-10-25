import random
import torch
from torch.utils.data.sampler import Sampler
import numpy as np


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
    def __init__(self, data_source, meta_data=None, reduction=20, num_samples_cls=3):
        random.seed(0)
        torch.manual_seed(0)
        self.data_source = data_source
        self.meta_data = meta_data
        self.reduction = reduction

        self.epoch = 0
        num_classes = 80

        # 类别选择的random Sampler
        self.class_iter = RandomCycleIter(list(range(num_classes)))

        self.cls_data_list = self.meta_data["cls_data_list"]
        self.gt_labels = self.meta_data["gt_labels"]

        self.num_classes = len(self.cls_data_list)

        # 对于每一个类种数据选择的random Sampler
        self.data_iter_list = [RandomCycleIter(x) for x in self.cls_data_list]

        self.num_samples = int(max([len(x) for x in self.cls_data_list])) * self.num_classes / self.reduction
        self.num_samples_cls = num_samples_cls

    def __iter__(self):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list, self.num_samples,
                                            self.num_samples_cls)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.data_source)

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
    from dataset import SDataset
    import os
    import mmcv
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms

    metadata = mmcv.load("../appendix/coco/longtail2017/metadata.pkl")
    root = "/data/coco/"

    data_path_train = f'{root}/train2017'
    instances_path_train = os.path.join(root, 'annotations/instances_train2017.json')

    train_dataset = SDataset(data_path_train,
                             instances_path_train,
                             metadata["img_ids"],
                             metadata["gt_labels"],
                             transform=transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                ]))

    sampler = ClassAwareSampler(train_dataset, metadata)
    loader = DataLoader(train_dataset, batch_size=32, shuffle=False, sampler=sampler, num_workers=2)

    results = torch.zeros(size =[80])
    for i, (imgs, targets) in enumerate(loader):
        results += torch.sum(targets, dim=0)
    print(results)


if __name__ == '__main__':
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