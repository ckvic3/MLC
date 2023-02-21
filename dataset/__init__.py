from .getDataset import getCOCODataset, getVocDataset,getDataloader
from .sampler import ClassAwareSampler
from .voc import VocDataset, copyDecouplingVocDataset 
from .coco import copyDecouplingCOCODataset,LTCOCODataset,CocoDataset

import mmcv
import numpy as np
@mmcv.register_handler('txt')
class TxtHandler1(mmcv.BaseFileHandler):

    def load_from_fileobj(self, file):
        return [line.strip('\n') for line in file.readlines()]

    def dump_to_fileobj(self, obj, file):
        for i in range(len(obj)):
            file.write(str(obj[i])+'\n')

    def dump_to_str(self, obj, **kwargs):
        return str(obj)


def multi2single(idsFile, classFreqFile):
    """

    Args:
        idsFile: 存储的是img id，顺序与class freq中gt_labels一致
        classFreqFile: dict 格式，我们需要其中的 "gt_labels"

    Returns:
        single_ids      :   转换后单标签图片对应 原始标记中的id
        masks           :   单标签图片对应的掩码
    """
    multi_ids = mmcv.list_from_file(idsFile)
    multi_labels = np.stack(mmcv.load(classFreqFile)['gt_labels'])
    single_ids = []
    masks = []

    for i, id in enumerate(multi_ids):
        labels = multi_labels[i,:]
        for j in range(multi_labels.shape[1]):
            if multi_labels[i, j] == 1:
                single_ids.append(id)
                complement_mask = 1 - labels  # M = (1 - Y) ∩ Y^i
                complement_mask[j] = 1      # M = (1 - Y) ∩ Y^i
                masks.append(complement_mask)
    masks = np.stack(masks)
    return single_ids, masks