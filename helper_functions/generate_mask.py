"""

将多标签数据分离成单标签数据，
new_img_id   ,保存新的img_id对应的原始的img_id

masks ,  每个新的样本对应的gt 掩码向量

"""

import mmcv
import numpy as np

# from custom import CustomDataset
#
# # root = "/data/coco/train2017/"
# # ann_file = "/data/coco/annotations/instances_train2017.json"
# LT_ann_file = "/home/pengpeng/DistributionBalancedLoss/appendix/coco/longtail2017/img_id.pkl"
# # dataset = CustomDataset(root, ann_file, LT_ann_file)
#
# class_freq_file = "/home/pengpeng/ASL/appendix/coco/longtail2017/class_freq.pkl"
# condition_prob = mmcv.load(class_freq_file)['condition_prob']
#
# class_freq = mmcv.load(class_freq_file)['class_freq']
#
# IncIndexes = np.argsort(class_freq)
# print("increment indexes",IncIndexes)
#
# gt_labels = mmcv.load(class_freq_file)['gt_labels']
#
# gt_labels = np.stack(gt_labels)
# print(gt_labels.sum().sum())
# # print("shape of gt labels :",np.shape(gt_labels))
# # exit()
#
# img_ids = mmcv.list_from_file(LT_ann_file)
#
# i = 0
# new_img_ids = []
# masks = []
# single_gt_labels = []
# real_gt_labels = []
# for i,img_id in enumerate(img_ids):
#     flag = False
#     for j in IncIndexes:
#         original_mask = 1 - gt_labels[i,:]
#         # 减少采样后负样本数量, 将mask 与 样本数量较少的样本绑定，其余分离的新样本，只包含正样本标签
#         mask = np.zeros(shape=[80])
#         if gt_labels[i,j] == 1:
#             new_img_ids.append(img_id)
#             real_gt_labels.append(gt_labels[i, :])
#             if flag:
#                 mask[j] = 1
#                 masks.append(mask)
#                 single_gt_labels.append(gt_labels[i,:] * mask)
#
#             else:
#                 flag = True
#                 original_mask[j] = 1
#                 masks.append(original_mask)
#                 single_gt_labels.append(gt_labels[i, :] * original_mask)
#
#
# new_img_ids = np.stack(new_img_ids)
# masks = np.stack(masks)
# single_gt_labels = np.stack(single_gt_labels)
#
# print("分离后 数据量: ", len(new_img_ids))
# np.save("./reduce_neg_new_img_ids.npy",new_img_ids)
# np.save("./reduce_neg_masks.npy",masks)
# np.save("./reduce_neg_single_gt_labels",single_gt_labels)
#
#
#
#
#
# total_num = np.sum(masks,axis=0)
# print(total_num)
#
# real_gt_labels = np.stack(real_gt_labels)
# pos_freq = np.sum(single_gt_labels,axis=0,keepdims=False)
#
# neg_freq_ = single_gt_labels.shape[0] - pos_freq
# neg_freq = single_gt_labels.shape[0] - np.sum(real_gt_labels,axis=0,keepdims=False)
#
# print("pos freq")
# print(pos_freq)
# # 把mask考虑在内时的mask
# print("neg freq")
# print(neg_freq)
# # 不考虑mask时的neg freq
# print("neg_freq_")
# print(neg_freq_)
#
# exit()
# pos_weight = np.max(pos_freq) / pos_freq
# neg_weight = np.max(pos_freq) / neg_freq
# print("pos weight")
# print(pos_weight)
# print("neg weight")
# print(neg_weight)
#
# np.save("./pos_weight.npy",pos_weight)
# np.save("./neg_weight.npy",neg_weight)



def multi2single(ann_file, class_freq_file):
    """

    Args:
        class_freq_file: 文件路径，对应mmcv 格式文件， 文件内容为 id 对应的multi label 标签
        ann_file: 文件路径，对应 mmcv格式文件， 文件内容为LT数据 ids

    Returns:
        single_ids      :   转换后单标签图片对应 原始标记中的id
        masks           :   单标签图片对应的掩码
    """
    multi_ids = mmcv.list_from_file(ann_file)
    multi_labels = np.stack(mmcv.load(class_freq_file)['gt_labels'])
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
    return single_ids, masks

