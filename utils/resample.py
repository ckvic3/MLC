"""
TODO
1、原始的类别分布
2、将每一张图片的标签重新组合，组成多种label，称为resplit. 计算类别分布
3、使用 resplit 后的图片resample，计算类别分布

"""
import random
import numpy as np
import mmcv


# metadata keys : 'gt_labels', 'class_freq', 'neg_class_freq',
# 'condition_prob', 'img_ids', 'cls_data_list', 'class_split'

metadata = mmcv.load("/home/pengpeng/MLC/appendix/coco/longtail2017/metadata.pkl")

splits = mmcv.load("../longtail2017/class_freq.pkl")

exit()


def dataSplits(groups=None, save_path="/home/pengpeng/ASL/appendix/coco/IncLearning/groups_id.pkl"):
    metadata = mmcv.load("/home/pengpeng/MLC/appendix/coco/longtail2017/metadata.pkl")
    img_ids = metadata["img_ids"]
    gt_labels = np.array(metadata["gt_labels"])

    num_classes = gt_labels.shape[1]
    num_groups = len(groups)

    ids = [[] for i in range(num_groups)]
    for i, group in enumerate(groups):
        mask = np.zeros(num_classes)
        mask[group] = 1
        temp_labels = gt_labels * mask
        temp_sum = np.sum(temp_labels, axis=1)
        for j in range(temp_sum.shape[0]):
            if temp_sum[j] > 0:
                ids[i].append(img_ids[j])

    mmcv.dump(ids, file=save_path)



metadata = mmcv.load("/home/pengpeng/MLC/appendix/coco/longtail2017/metadata.pkl")
groups = metadata["class_split"]

dataSplits(groups)

group_ids = mmcv.load("/home/pengpeng/ASL/appendix/coco/IncLearning/groups_id.pkl")

for group_id in group_ids:
    print(len(group_id))
exit()

pkl_data = mmcv.load("../appendix/coco/longtail2017/class_freq.pkl")

img_ids = np.load("../appendix/coco/longtail2017/img_ids.npy")

splits = mmcv.load("../appendix/coco/longtail2017/class_split.pkl")

print(splits)
exit()
head = list(splits["head"])
middle = list(splits["middle"])
tail = list(splits["tail"])

gt_labels = np.array(pkl_data["gt_labels"])
# 原始数据的数量
print(gt_labels.shape[0])

metadata = mmcv.load("../appendix/coco/longtail2017/metadata.pkl")
img_ids = []
for i in range(80):
    img_id = []
    for j in range(gt_labels.shape[0]):
        if gt_labels[j, i] == 1:
            img_id.append(j)
    img_ids.append(img_id)
metadata["cls_data_list"] = img_ids

mmcv.dump(metadata, "../appendix/coco/longtail2017/metadata.pkl")

exit()

class_freq = pkl_data["class_freq"]
print(class_freq)

# 按照数量从小到大的类别下标排序
# ret_indexes = np.array(class_freq).argsort()
# print(ret_indexes)

# 将每一张图片的标签重新组合，组成多种 label
# 不能将每一张图片都进行拆分，因为

new_labels = []
max_sum = 0
for i in range(gt_labels.shape[0]):
    if np.sum(gt_labels[i, head]) != 0 and np.sum(gt_labels[i, tail]) != 0:
        for j in head:
            if gt_labels[i, j] == 1:
                new_gt = np.copy(gt_labels[i])
                new_gt[head] = 0
                new_gt[j] = 1
                new_labels.append(new_gt)
    else:
        new_labels.append(gt_labels[i])

data = {}
new_labels = np.array(new_labels)
data["gt_labels"] = new_labels

# 计算新标签中每个类别的数量
new_class_freq = new_labels.sum(axis=0)
# print(new_class_freq)
data["class_freq"] = new_class_freq
print(new_labels.shape[0])

# 为每个类别筛选其对应的新labels
img_ids = []
for i in range(80):
    img_id = []
    for j in range(new_labels.shape[0]):
        if new_labels[j, i] == 1:
            img_id.append(j)
    img_ids.append(img_id)

data["img_ids"] = img_ids

mmcv.dump(data, "./seperate_tailAndHead.pkl")

random.seed(299)
IMGNUM = 200

result = np.zeros([80])

for i in range(80):
    img_id = img_ids[i]
    length = len(img_id)
    # print("length",len(img_id))
    for j in range(IMGNUM):
        sel = random.randint(0, length - 1)
        result += new_labels[img_id[sel]]

print(result)
