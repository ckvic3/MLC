#  用于绘制样本数量与对应Ap分数展示图
# 
import numpy as np
import matplotlib.pyplot as plt
import mmcv 

plt.figure('deepinout.com 极客笔记',figsize=[15,4])
# plt.axis([0, 81, 0, 1300])
plt.axis('on')
plt.grid(True)

#定义显示的数据


metadata = mmcv.load("/home/pengpeng/MLC/appendix/coco/longtail2017/class_freq.pkl")
print(metadata["class_freq"])
class_freq = np.array(metadata["class_freq"])

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

x = range(1,class_freq.shape[0]+1)

#显示index与数量曲线
plt.yscale('log')
ln1 = plt.plot(x, class_freq, 'b', label = 'sample number')
plt.scatter(x, class_freq, color = 'b', marker = 's')

plt.xlabel('class index')
plt.ylabel('sample number', color = 'b')
plt.tick_params(axis = 'y', labelcolor = 'b')
#图例
plt.legend()


#显示index与aps曲线
ax2 = plt.twinx()
plt.yscale('linear')
ln2 = ax2.plot(x, aps, 'r', label = 'Ap')
ax2.scatter(x, aps, color = 'r', marker = '*', s = 50)
ax2.set_ylabel('Ap', color = 'r')
plt.tick_params(axis = 'y', labelcolor = 'r')
#图例
ln = ln1 + ln2 
labels = [l.get_label() for l in ln]
plt.legend(ln,labels,loc=0)

plt.savefig("./example.jpg")





