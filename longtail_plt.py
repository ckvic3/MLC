import mmcv 
import numpy as np 
from scipy.interpolate import make_interp_spline,interp1d
import matplotlib.pyplot as plt 
metadata = mmcv.load("/home/pengpeng/MLC/appendix/coco/longtail2017/class_freq.pkl")
print(metadata["class_freq"])
class_freq = metadata["class_freq"]
x = range(class_freq.shape[0]) 
x = np.array(x) + 1
y = - np.sort(-class_freq) 

model = interp1d(x,y,kind="cubic")

xs = np.linspace(1,80,1000)
ys = model(xs)
plt.figure("long tail")

plt.plot(xs,ys)
plt.xlabel("Sorted class index")
plt.ylabel("Number of images")
plt.savefig("./longtail.jpg")