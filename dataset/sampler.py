import random
import torch
from torch.utils.data import Sampler
import mmcv
from .coco import copyDecouplingCOCODataset
class ClassAwareSampler(Sampler):

    def __init__(self, data_source, num_samples_cls=3, num_classes=80, clsDataListFile=None):
        random.seed(0)
        torch.manual_seed(0)
        self.epoch = 0
        # a generator, generate next sampling class
        self.class_iter = RandomCycleIter(range(num_classes))
        # cls_data_list = [list() for _ in range(num_classes)]
        '''
        labels = [ i for i in range(num_classes)]
        for i, label in enumerate(labels):
            cls_data_list[label].append(i)'''

        # 获取每个类别对应的img_id和真实标签
        data = mmcv.load(clsDataListFile)
        self.cls_data_list = data['cls_data_list']

        self.num_classes = len(self.cls_data_list)
        self.data_iter_list = [RandomCycleIter(x) for x in self.cls_data_list] # repeated
        self.num_samples = len(data_source)
        if type(data_source) == copyDecouplingCOCODataset:
            self.num_samples = 1909

        # 针对选中的类别每次采样样本数量
        self.num_samples_cls = num_samples_cls
        print('>>> Class Aware Sampler Built!')

    def __iter__(self):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list,
                                            self.num_samples, self.num_samples_cls)

    def __len__(self):
        return self.num_samples

    def set_epoch(self,  epoch):
        self.epoch = epoch

class RandomCycleIter:
    """
    select a random item from data_list
    """
    def __init__(self, data_list, test_mode=False):
        """

        Args:
            data_list:
            test_mode:
        """
        self.data_list = list(data_list)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1

        # 完成一次遍历后，重新打乱data list的顺序
        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)

        return self.data_list[self.i]


def class_aware_sample_generator(cls_iter, data_iter_list, n, num_samples_cls=1):
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

