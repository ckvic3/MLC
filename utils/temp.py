# import numpy as np
# temp = [[0,0,0,3,0,6,0,0],
#         [0,0,0,0,0,0,6,0],
#         [0,2,0,0,0,0,0,0],
#         [0,0,0,0,0,0,0,0],
#         [6,0,0,0,0,0,0,0],
#         [0,2,6,0,0,0,0,0],
#         [0,0,0,0,0,0,0,6],
#         [0,2,0,3,6,0,0,0]
#         ]

# temp = np.array(temp)

# init = np.array([1,1,1,1,1,1,1,1])

# init = np.reshape(init,[-1,1])

# div1 = 48
# div2 = 48 * 6

# v1 = np.matmul(temp, init)
# print("version1",v1)
# print(np.sum(v1))
# v2 = np.matmul(temp, v1)
# print("version2", v2)
# print(np.sum(v2))
import mmcv
import numpy as np

if __name__ == "__main__":
    path = "/home/pengpeng/MLC/appendix/VOCdevkit/longtail2012/class_freq.pkl"
    metadata = mmcv.load(path)
    print(len(metadata['gt_labels']))
    print(metadata['class_freq'])

    gt_labels = np.array(metadata['gt_labels'])
    print(len(gt_labels))
    for i in range(20):
        data_list = []
        for j in range(gt_labels.shape[0]):
            if gt_labels[j][i] == 1:
                data_list.append(j)
        cls_data_list.append(data_list)
    metadata['cls_data_list'] = cls_data_list

    mmcv.dump(metadata,path)