from pyexpat import model
from turtle import forward
from matplotlib.transforms import TransformedPatchPath
import mmcv
import numpy as np
from sqlalchemy import true 
import torch
import torch.nn as nn
import os
from tqdm import trange
from datasets.coco import LTCOCO, m2sCoco
from src.sampler import m2sSample
import torchvision.transforms as transforms
from datasets import getVocDataset

class sampleModel(nn.Module):
    def __init__(self):
        super(sampleModel,self).__init__()
        self.weight = torch.nn.Parameter(data=torch.ones(size=[80]),requires_grad=True).double()
        self.weight.data.fill_(0.25)
        self.weight.retain_grad()

    def forward(self, x):
        w = torch.abs(self.weight)
        return torch.matmul(x,w)

best_weight = \
       [1.3253, 0.1316, 1.5482, 0.8394, 1.1388, 0.4943, 1.2010, 0.5193, 1.8051,
        0.3984, 0.8000, 0.8884, 1.1275, 0.5025, 0.4120, 1.3665, 0.8655, 0.9255,
        0.8463, 1.3328, 1.6232, 2.1883, 1.0369, 1.5887, 0.0429, 0.3653, 0.2895,
        0.8633, 0.2885, 0.8661, 1.3361, 1.4456, 0.5675, 1.1437, 0.9998, 0.1425,
        1.0989, 1.4416, 0.9967, 0.7632, 0.2116, 0.0214, 0.6468, 0.4866, 0.2145,
        0.4929, 0.8508, 0.9722, 0.6691, 1.1246, 2.0541, 1.2418, 1.5400, 0.6492,
        1.3721, 0.9192, 0.8657, 0.2156, 0.4128, 1.2945, 0.3095, 2.0670, 0.8543,
        0.3909, 0.6416, 0.6638, 0.3767, 0.3785, 0.0707, 0.3477, 0.2100, 1.0459,
        0.0156, 0.3921, 0.0342, 1.0371, 1.5938, 1.1127, 0.3643, 1.3109]

def computeWeight():
    # metadata = mmcv.load("/home/pengpeng/MLC/appendix/coco/longtail2017/class_freq.pkl")
    metadata = mmcv.load("/home/pengpeng/MLC/appendix/VOCdevkit/longtail2012/class_freq.pkl")

    condition_prob = metadata["condition_prob"]


    weight =torch.ones(size=[19]).double().requires_grad_(True)
    weight.data.fill_(0.25)

    condition_prob = np.array(condition_prob)
    condition_prob = np.delete(condition_prob,14,0)
    condition_prob = np.delete(condition_prob,14,1)
    p = np.array(condition_prob).transpose()
    p = torch.from_numpy(p)

    min_loss = 100
    best_weight = None
    optimizer = torch.optim.SGD(params={weight}, lr=1e-6, weight_decay=1e-4, momentum=0.9)

    for i in range(10000):
        w = torch.abs(weight)
        out = torch.matmul(p,w)
        loss = torch.var(out/out.max())
        
        loss.backward()        
        optimizer.step()
        print(loss)
        if loss < min_loss:
            min_loss = loss
            best_weight = weight

    print("min_loss",min_loss)
    print("best weight:\n",torch.abs(best_weight))    
    
    print("class aware:")
    original = torch.matmul(p,torch.ones(size=[19]).double())
    result = original / original.max()
    print(result)
    print("var:",torch.var(result))

    print("optim aware:")
    original = torch.matmul(p,torch.abs(best_weight))
    result = original / original.max()
    print(result)
    print("var:",torch.var(result))

    return torch.abs(best_weight)


def getMlDatasetweights(dataset, weights_save_path="/home/pengpeng/ASL/appendix/coco/longtail2017/ml_sample_weights.npy"):
    """
        获取多标签数据集尽量 class balance 采样时每个样本的采样概率。
        Args:
            dataset: ml dataset
        Returns:
            sample_weights
        """
    if os.path.exists(weights_save_path):
        sample_weigths = np.load(weights_save_path)
    else:
        weights = np.array(best_weight)
        sample_weigths = []
        for i in trange(len(dataset)):
            _, target= dataset.__getitem__(i)
            target = target.numpy()
            # print("target",target.shape)
            # print("weights:",weights.shape)
            weight = (weights * target).sum()
            sample_weigths.append(weight)
        sample_weigths = np.array(sample_weigths)
        np.save(weights_save_path, sample_weigths)
    return sample_weigths



def getResult():
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    trainTransform = transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalization
                                    ])

    root = "/data/coco"
    data_path_train = f'{root}/train2017'
    instances_path_train = os.path.join(
        root, 'annotations/instances_train2017.json')
    dataset = LTCOCO(data_path_train, instances_path_train,
                    LT_ann_file="/home/pengpeng/MLC/appendix/coco/longtail2017/img_id.pkl",
                    transform=trainTransform)
    
    sampleWeights = getMlDatasetweights(dataset)

    print(sampleWeights.shape)

    sampler = m2sSample(data_source=dataset,reduction=8,num_classes=80,weights=sampleWeights)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False,
        sampler=sampler,
        num_workers=4, pin_memory=True)

    counts = torch.zeros(80)
    for i, (inputData, target) in enumerate(loader):
        counts += target.sum(dim=0)
        if i % 100 == 0:
            print(i)
    print(counts)

from src.sampler import m2sSample,ClassAwareSampler
if __name__ == '__main__':
    # computeWeight()
    train_dataset, val_dataset = getVocDataset(use_m2s=False,image_size=224)
    weights_save_path = "/home/pengpeng/MLC/appendix/VOCdevkit/longtail2012/sample_weights.npy"
    class_freq_file = "/home/pengpeng/MLC/appendix/VOCdevkit/longtail2012/class_freq.pkl"
    SPLIT_GROUP = [[4,6,8,10,14,15],
        [1,7,11,13,17,19],
        [0,2,3,5,9,12,16,18]]
    num_classes = 20
    metadata = mmcv.load("/home/pengpeng/MLC/appendix/VOCdevkit/longtail2012/class_freq.pkl")
    shuffle = False
    sampler = ClassAwareSampler(train_dataset,metadata)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=shuffle,
        sampler = sampler,
        num_workers=4, pin_memory=True)
    
    counts = torch.zeros(20)
    for i, (imgs,targets) in enumerate(train_loader):
        counts += targets.sum(dim=0)
    
    print(counts)