import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
import xml.etree.ElementTree as ET
import collections
import torch 
import mmcv 
import torchvision.transforms as transforms

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

VOC_CAT_MAPS = {"aeroplane": 0, "bicycle": 1, "bird": 2, "boat": 3, "bottle": 4, "bus": 5,
                "car": 6, "cat": 7, "chair": 8, "cow": 9, "diningtable": 10, "dog": 11,
                "horse": 12, "motorbike": 13, "person": 14, "pottedplant": 15, "sheep": 16,
                "sofa": 17, "train": 18, "tvmonitor": 19}

class VocDataset(Dataset):
    def __init__(self,imgPath,annoPath,LtAnnFile,transform=None) -> None:
        self.imgPath = imgPath
        self.annoPath = annoPath
        self.transform = transform
        if LtAnnFile is not None:
            self.imgIds = mmcv.list_from_file(LtAnnFile)

    def __getitem__(self, idx: int):
        assert idx < len(self.imgIds)
        
        img = self.get_image_(idx)
        target = self.get_label_(idx)
        
        return img, target, target

    def get_image_(self,idx):
        imgFileName = os.path.join(self.imgPath, str(self.imgIds[idx]) + ".jpg")
        assert os.path.exists(imgFileName)
        img = Image.open(imgFileName).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img
    
    def get_label_(self,idx):
        annFileName = os.path.join(self.annoPath, str(self.imgIds[idx]) + '.xml')
        assert os.path.exists(annFileName)
        
        objects = self.parse_voc_xml(ET.parse(annFileName).getroot())['annotation']['object']
        target = np.zeros(shape=[20])
        for object in objects:
            if object['name'] not in VOC_CAT_MAPS.keys():
                print(object['name'])
                raise IndexError
            else:
                target[VOC_CAT_MAPS[object['name']]] = 1

        return target 
    
    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == 'annotation':
                def_dic['object'] = [def_dic['object']]
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict
    
    def __len__(self) -> int:
        return len(self.imgIds)

class copyDecouplingVocDataset(VocDataset):
    def __init__(self, root, annoFile, singleLabelFile,maskFile,transform=None) -> None:
        super(copyDecouplingVocDataset,self).__init__(root, annoFile, None,transform)
        self.imgIds = mmcv.load(singleLabelFile)
        self.masks = np.load(maskFile)
     
    def __getitem__(self, idx: int):
        assert idx < len(self.imgIds)
        img = self.get_image_(idx)
        target = self.get_label_(idx)
        mask = self.masks[idx]

        return img,target,mask
    
if __name__ == '__main__':
    root = "/data/VOCdevkit/VOC2012/JPEGImages"
    annoFile = "/data/VOCdevkit/VOC2012/Annotations/"
    imgIdFile = "/home/pengpeng/MLC/appendix/VOCdevkit/longtail2012/img_id.txt"
    dataset = VocDataset(root,annoFile,imgIdFile)
    
    # sums = np.zeros(shape=[20])
    # for i in range(dataset.__len__()):
    #     _,target = dataset.__getitem__(i)
    #     sums += target
    # print(sums  )
    
    m2sImgIdFile = "/home/pengpeng/MLC/appendix/VOCdevkit/longtail2012/voc_m2s_ids.pkl"
    maskFile = "/home/pengpeng/MLC/appendix/VOCdevkit/longtail2012/masks.npy"
    m2sDataset = copyDecouplingVocDataset(root,annoFile,m2sImgIdFile,maskFile,transform=transforms.Compose([
                               transforms.Resize((224, 224)),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                           ]))

    # sums = np.zeros(shape=[20])
    # for i in range(m2sDataset.__len__()):
    #     _,target,mask = m2sDataset.__getitem__(i)
    #     sums+= target*mask
    # print(sums  )
    
    loader = torch.utils.data.DataLoader(
        m2sDataset, batch_size=32, shuffle=True,
        num_workers=8, pin_memory=True)

    for (i,(imgs,targets,masks)) in enumerate(loader):
        print(imgs.shape)
        print(targets.shape)
        print(masks.shape)

        print(type(imgs))
        print(type(targets))
        print(type(masks))

