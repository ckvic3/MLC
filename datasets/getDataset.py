import torchvision.transforms as transforms
import os
from .coco import CocoDetection,LTCOCO,m2sCoco
from .voc import VocDataset, m2sVoc 

def getCOCODataset(use_m2s:bool, image_size):
    """
    use_m2s: 是否使用multi 2 single 数据预处理
    
    """
    root = "/data/coco"
    data_path_train = f'{root}/train2017'
    data_path_val = f'{root}/val2017'

    instances_path_train = os.path.join(
        root, 'annotations/instances_train2017.json')
    instances_path_val = os.path.join(
        root, 'annotations/instances_val2017.json')

    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    trainTransform = transforms.Compose([
                                        transforms.Resize((image_size, image_size)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalization
                                    ])
    if use_m2s:
        meta_path = "/home/pengpeng/ASL/appendix/coco/coco_train_2017.json"
        m2s_path = "/home/pengpeng/MLC/appendix/coco/longtail2017/m2s_img_ids.npy"
        mask_path = "/home/pengpeng/MLC/appendix/coco/longtail2017/m2s_masks.npy"
        train_dataset = m2sCoco(data_path_train, instances_path_train, meta_path, 
            transform=trainTransform,
            m2s_path=m2s_path, mask_path=mask_path)
    else:
        train_dataset = LTCOCO(data_path_train, instances_path_train,
                           LT_ann_file="/home/pengpeng/MLC/appendix/coco/longtail2017/img_id.pkl",
                           transform=trainTransform)

    val_dataset = CocoDetection(data_path_val,
                                instances_path_val,
                                transforms.Compose([
                                    transforms.Resize(
                                    (image_size, image_size)),
                                    transforms.ToTensor(),
                                    normalization
                                ]))

    print("len(val_dataset)): ", len(val_dataset))
    print("len(train_dataset)): ", len(train_dataset))
    return train_dataset,val_dataset



def getVocDataset(use_m2s:bool,image_size:int):
    root = "/data/VOCdevkit/VOC2012/JPEGImages"
    annoFile = "/data/VOCdevkit/VOC2012/Annotations/"
    imgIdFile = "/home/pengpeng/MLC/appendix/VOCdevkit/longtail2012/img_id.txt"

    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    trainTransform = transforms.Compose([
                                        transforms.Resize((image_size, image_size)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalization
                                    ])

    if use_m2s:
        m2sImgIdFile = "/home/pengpeng/MLC/appendix/VOCdevkit/longtail2012/voc_m2s_ids.pkl"
        maskFile = "/home/pengpeng/MLC/appendix/VOCdevkit/longtail2012/masks.npy"
        train_dataset = m2sVoc(root,annoFile,m2sImgIdFile,maskFile,transform=trainTransform)
    else:
        train_dataset = VocDataset(root,annoFile,imgIdFile,transform=trainTransform)
    
    root = "/data/VOCdevkit/VOC2007/JPEGImages"
    annoFile = "/data/VOCdevkit/VOC2007/Annotations/"
    imgIdFile = "/home/pengpeng/MLC/appendix/VOCdevkit/longtail2012/test.txt"
    test_dataset = VocDataset(root,annoFile,imgIdFile,transform=trainTransform)
    
    print("len(val_dataset)): ", len(test_dataset))
    print("len(train_dataset)): ", len(train_dataset))
    return train_dataset, test_dataset


if __name__ =='__main__':
    train_dataset, test_dataset = getVocDataset(True,224)
    print(train_dataset.__getitem__(2))
    print(test_dataset.__getitem__(2))