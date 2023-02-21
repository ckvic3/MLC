
import mmcv
import torch
import time
import numpy as np 
# backbone=dict(
#         type='ResNet',
#         depth=50,
#         num_stages=4,
#         out_indices=(0, 1, 2, 3),
#         frozen_stages=1,
#         style='pytorch')


# model = ResNet(depth=50,num_stages=4,
#                  strides=(1, 2, 2, 2),
#                  dilations=(1, 1, 1, 1),
#                  out_indices=(0, 1, 2, 3),
#                  style='pytorch',
#                  frozen_stages=-1,
#                  conv_cfg=None,
#                  norm_cfg=dict(type='BN', requires_grad=True),
#                  norm_eval=True,
#                  gen_attention=None,
#                  )
# x = torch.rand(size=[3, 3, 224, 224])
# begin = time.time()
# out = model(x)

# print(out[-1].shape)
# print("time",time.time() - begin)

SPLIT_GROUP = [[0, 2, 24, 26, 39, 41, 42, 43, 44, 45, 56, 57, 58, 60, 62, 63, 67, 69, 71, 72, 73, 75],
               [1, 3, 5, 7, 8, 9, 13, 15, 16, 25, 27, 28, 32, 34, 35, 38, 40, 46, 47, 48, 49, 50, 51,
                53, 55, 59, 61, 64, 65, 66, 68, 74, 77],
               [4, 6, 10, 11, 12, 14, 17, 18, 19, 20, 21, 22, 23, 29, 30, 31, 33, 36, 37, 52, 54, 70, 76, 78, 79]]


# model = GroupModule(super_labels=SPLIT_GROUP,Pretrained="torchvision://resnet50")
# model.init_weights()

# model = GroupModel(SPLIT_GROUP)


# x = torch.rand(size=[3,3,224,224])

# out = model(x)
# print(out)
# print(out.shape)
# print(model)


# 验证损失函数的 mean 和 sum 间的区别  sum = mean * N * C 
# N 为batch size, C 为类别数量
# import torch.nn as nn
# lossSum = nn.BCEWithLogitsLoss(reduction='sum')
# lossMean = nn.BCEWithLogitsLoss(reduction='mean')
# input = torch.rand(2,3)
# target = torch.randint(low=0,high=2,size=[2,3]).float()

# sum = lossSum(input,target)
# mean = lossMean(input,target)

# print("sum ",sum)
# print("mean ",mean)


aps = [0.9076417,0.40466112, 0.57292066, 0.69730023,0.24055795, 0.69852918,
 0.79193577, 0.43189204, 0.24715809, 0.46991253, 0.17767229, 0.10536154,
 0.08271065, 0.3028756,  0.06495468, 0.77390546, 0.57378328, 0.27811052,
 0.16478476, 0.2109011,  0.46888465, 0.13942208, 0.68941948, 0.12445179,
 0.24312972, 0.4431022, 0.29850417, 0.49189704, 0.06682989, 0.05422553,
 0.73443529, 0.23654865, 0.39838243, 0.09504582, 0.41096359, 0.75140433,
 0.05908103, 0.46011888, 0.74858912, 0.35371342, 0.34793669, 0.43038026,
 0.44869623, 0.35117144, 0.31585869, 0.43763329, 0.31503088, 0.35109601,
 0.32946983, 0.46871237, 0.7063657,  0.241528,   0.14214727, 0.67007308,
 0.10097103, 0.38939334, 0.47818625, 0.58974612, 0.32532845, 0.54875004,
 0.60775282, 0.69544302, 0.71121639, 0.65500935, 0.67561965, 0.39390397,
 0.71597775, 0.24747042, 0.3663109,  0.56221022, 0.08957364, 0.68964488,
 0.4997156,  0.40507691, 0.60950897, 0.48995659, 0.00918762, 0.28922765,
 0.00377047, 0.05011499]


aps = np.array(aps)
result = np.argsort(aps)


result = result.tolist()
print("result:",result)

import mmcv 


metadata = mmcv.load("/home/pengpeng/MLC/appendix/coco/longtail2017/class_freq.pkl")
print(metadata["class_freq"])

class_freq = np.array(metadata["class_freq"])
class_split = mmcv.load("/home/pengpeng/MLC/appendix/coco/longtail2017/class_split.pkl")

index = np.argsort(class_freq).tolist()
print(index)

for key in class_split.keys():
    sum = 0 
    for idx in class_split[key]:
        sum += class_freq[idx]
    print(key,":",sum)

condition_prob = metadata["condition_prob"]
condition_prob = condition_prob.transpose()

index = np.argsort(class_freq).tolist()
print("before resampling class freq sort:\n",index)

porb_after_class_aware_sample = np.matmul(condition_prob, np.ones(shape=[80,1])).reshape(80)
after_sort_index = porb_after_class_aware_sample.argsort().tolist()
print("after resampling class freq sort:\n",after_sort_index)

new_group = [sorted(after_sort_index[:25]),
sorted(after_sort_index[25:58]),
    sorted(after_sort_index[58:])]

print("after resamping split: \n",new_group)

# head middle tail 划分  
# {'head': {0, 2, 24, 26, 39, 41, 42, 43, 44, 45, 56, 57, 58, 60, 62, 63, 67, 69, 71, 72, 73, 75},  22种
# 'middle': {1, 3, 5, 7, 8, 9, 13, 15, 16, 25, 27, 28, 32, 34, 35, 38, 40, 46, 47, 48, 49, 50, 51, 
# 53, 55, 59, 61, 64, 65, 66, 68, 74, 77}, 33种
# 'tail': {4, 6, 10, 11, 12, 14, 17, 18, 19, 20, 21, 22, 23, 29, 30, 31, 33, 36, 37, 52, 54, 70, 76, 78, 79}} 25种



def generateGroupsBySampleNumber(GROUP_NUMBER, class_freq):
    """
    parameters:

    GROUP_NUMBER: 期待分成的组
    class_freq: 各类别样本数量

    """
    index = np.argsort(class_freq).tolist()
    length = len(index)

    groups = []
    begin = 0

    for i in range(GROUP_NUMBER):
        if i == GROUP_NUMBER - 1:
            tempGroup = index[begin:]
            tempGroup.sort()
            groups.append(tempGroup)
        else:
            tempGroup = index[begin:begin+length//GROUP_NUMBER]
            tempGroup.sort()
            groups.append(tempGroup) 
            begin += length // GROUP_NUMBER

    print("sort result \n",groups)


    
# ## super parameter
# GROUP_NUMBER = 3

# print("语义分组")    
# generateGroupsBySampleNumber(GROUP_NUMBER, aps)


# # [70,78,24,26,44,76,47,13,67,73,12,58,43,7,79,45,39,48,28,75,41,60,51,52,42,1,11,40,56,49,14,65,72,74,68,57,55,27,54,25]
# # [31,10,9,46,66,77,64,69,5,62,32,59,2,8,16,63,50,71,19,3,17,30,29,53,15,34,35,37,18,6,4,33,21,36,61,20,38,0,22,23]

# print("class_freq:\n",class_freq)


# class_freq = mmcv.load("/home/pengpeng/MLC/appendix/VOCdevkit/longtail2012/class_freq.pkl")["class_freq"]
# print("class_freq", class_freq)


# from sklearn.cluster import KMeans
# # class_freq =  np.clip(class_freq,a_min=6,a_max=1000)
# CLUSTER_NUMBER = 6

# model = KMeans(n_clusters=CLUSTER_NUMBER,max_iter=1000)

# # aps = [0.91567702, 0.85883055, 0.67675734, 0.83529864, 0.50828806, 0.72878913,
# #  0.87249976, 0.90105826, 0.63545981, 0.26306341, 0.78156195, 0.82306364,
# #  0.72505761, 0.82967198, 0.89269407, 0.60429445, 0.43028978, 0.67118919,
# #  0.8394131,  0.79167786]
# # COCO APS 
# aps = [0.9076417,0.40466112, 0.57292066, 0.69730023,0.24055795, 0.69852918,
#  0.79193577, 0.43189204, 0.24715809, 0.46991253, 0.17767229, 0.10536154,
#  0.08271065, 0.3028756,  0.06495468, 0.77390546, 0.57378328, 0.27811052,
#  0.16478476, 0.2109011,  0.46888465, 0.13942208, 0.68941948, 0.12445179,
#  0.24312972, 0.4431022, 0.29850417, 0.49189704, 0.06682989, 0.05422553,
#  0.73443529, 0.23654865, 0.39838243, 0.09504582, 0.41096359, 0.75140433,
#  0.05908103, 0.46011888, 0.74858912, 0.35371342, 0.34793669, 0.43038026,
#  0.44869623, 0.35117144, 0.31585869, 0.43763329, 0.31503088, 0.35109601,
#  0.32946983, 0.46871237, 0.7063657,  0.241528,   0.14214727, 0.67007308,
#  0.10097103, 0.38939334, 0.47818625, 0.58974612, 0.32532845, 0.54875004,
#  0.60775282, 0.69544302, 0.71121639, 0.65500935, 0.67561965, 0.39390397,
#  0.71597775, 0.24747042, 0.3663109,  0.56221022, 0.08957364, 0.68964488,
#  0.4997156,  0.40507691, 0.60950897, 0.48995659, 0.00918762, 0.28922765,
#  0.00377047, 0.05011499]
# aps = np.array(aps)

# # labels = model.fit_predict(X=class_freq.reshape(-1,1))
# labels = model.fit_predict(X=aps.reshape(-1,1))

# groups = []
# for i in range(CLUSTER_NUMBER):
#     groups.append([])

# for i in range(aps.shape[0]):
#     groups[labels[i]].append(i)
# print(groups)





# import matplotlib.pyplot as plt
# x = range(class_freq.shape[0])

# plt.plot(x,class_freq,x,aps*class_freq.max())

# plt.savefig(fname="./freqAndaps.jpg")