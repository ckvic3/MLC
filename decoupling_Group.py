import os
import argparse
import torch
from models import GroupModel
from datasets.coco import CocoDetection
from src.helper_functions.dataset import m2sCoco
from src.helper_functions.helper_functions import mAP,seed_torch,AverageMeter,WarmUpLR
import torchvision.transforms as transforms
from datasets.getDataset import getCOCODataset,getVocDataset
from src.losses import AsymmetricLossOptimized as ASL
from src.losses import BCELoss as BCEWithLogitsLoss, FBLoss
from losses import ResampleLoss,FocalLoss
import numpy as np
from torch.optim import lr_scheduler
from src.sampler import m2sSample,ClassAwareSampler
import mmcv
from config import DIFFICULTY_GROUPS,SAMPLE_NUMBER_GROUPS,K_DIFFICULTY_GROUPS,K_SAMPLE_NUMBER_GROUPS
from voc_config import VOC_K_DIFFICULTY_GROUPS,VOC_K_SAMPLE_NUMBER_GROUPS

from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--lr', default=2e-2, type=float)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=224, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--epoch', default=80, type=int, required=False)
parser.add_argument('--num-classes', default=80, type=int)
parser.add_argument('--reduction', default=4, type=int)
parser.add_argument('--use-m2s',action="store_true")
parser.add_argument('--dataset-name',default="coco",type=str)
parser.add_argument('--output',type=str)

parser.add_argument('--use-rs',action="store_true",help="use classAware resampling")
parser.add_argument('--use-global',action="store_true",help="use global branch")
parser.add_argument('--use-group',action="store_true")
parser.add_argument('--loss-type',type=str,default="bce")

parser.add_argument('--device',type=int,default=0)
parser.add_argument('--group-number',type=int,default=0)
parser.add_argument('--group-by',type=str,default=None,help="n :number or d:difficulty")

parser.add_argument('--gamma-pos',default=0,type=int)
parser.add_argument('--gamma-neg',default=2,type=int)

parser.add_argument('--warm-iters',type=int,default=500)
parser.add_argument('--warm-radio',type=float,default=1.0 /3)

parser.add_argument('--model-path',default=None)
parser.add_argument('--config',type=str,default="")

args = parser.parse_args()

args.output = os.path.join("/data/pengpeng/checkpoint/",args.output)
os.makedirs(args.output, exist_ok=True)
writer = SummaryWriter(log_dir=args.output)

def main():
    if args.dataset_name == "coco":
        train_dataset, val_dataset = getCOCODataset(use_m2s=args.use_m2s,image_size=args.image_size)
        class_freq_file="/home/pengpeng/ASL/appendix/coco/longtail2017/class_freq.pkl"
        SPLIT_GROUP = [[0, 2, 24, 26, 39, 41, 42, 43, 44, 45, 56, 57, 58, 60, 62, 63, 67, 69, 71, 72, 73, 75],
               [1, 3, 5, 7, 8, 9, 13, 15, 16, 25, 27, 28, 32, 34, 35, 38, 40, 46, 47, 48, 49, 50, 51,
                53, 55, 59, 61, 64, 65, 66, 68, 74, 77],
               [4, 6, 10, 11, 12, 14, 17, 18, 19, 20, 21, 22, 23, 29, 30, 31, 33, 36, 37, 52, 54, 70, 76, 78, 79]]
        num_classes = 80
        metadata = mmcv.load("/home/pengpeng/MLC/appendix/coco/longtail2017/metadata.pkl")

    elif args.dataset_name == "voc":
        train_dataset, val_dataset = getVocDataset(use_m2s=args.use_m2s,image_size=args.image_size)

        class_freq_file = "/home/pengpeng/MLC/appendix/VOCdevkit/longtail2012/class_freq.pkl"
        SPLIT_GROUP = [[4,6,8,10,14,15],
            [1,7,11,13,17,19],
            [0,2,3,5,9,12,16,18]]
        num_classes = 20
        metadata = mmcv.load("/home/pengpeng/MLC/appendix/VOCdevkit/longtail2012/class_freq.pkl")

    if args.use_rs:
        shuffle = False
        sampler = ClassAwareSampler(train_dataset,metadata,args.reduction)
    else:
        sampler = None
        shuffle = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=shuffle,
        sampler=sampler,num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size * 4, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    used_groups = SPLIT_GROUP
    print("group by :", args.group_by)
    if args.group_by == "n":
        used_groups = SAMPLE_NUMBER_GROUPS[args.group_number - 1]
    elif args.group_by =="kn":
        if args.dataset_name == 'coco':
            used_groups = K_SAMPLE_NUMBER_GROUPS[args.group_number - 1]
        else:
            used_groups = VOC_K_SAMPLE_NUMBER_GROUPS[args.group_number - 1]
    elif args.group_by =="kd":
        if args.dataset_name == 'coco':
            used_groups = K_DIFFICULTY_GROUPS[args.group_number - 1]
        else:
            used_groups = VOC_K_DIFFICULTY_GROUPS[args.group_number - 1]
    elif args.group_by =="d":
        used_groups = DIFFICULTY_GROUPS[args.group_number - 1]
    
    print("group: \n",used_groups)
    model = GroupModel(groups=used_groups,num_classes=num_classes,use_global=args.use_global,use_group=args.use_group)
    
    # 加载权重
    # print("load backbone weight from simple model")
    # miss_key, unexpected_key = model.load_state_dict(state_dict=torch.load("/data/pengpeng/checkpoint/base_aug/model-highest.ckpt"),strict=False)
    # print(miss_key, unexpected_key)

    if args.model_path is not None and os.path.exists(args.model_path):
        model.load_state_dict(torch.load(f=args.model_path,map_location="cpu"),strict=False)
    model = model.cuda()
    print("model load finished!")
    
    # 验证
    # if args.model_path is not None and os.path.exists(args.model_path):
    #     validate(SPLIT_GROUP, used_groups, model, val_loader, 0, 0, 0)

    lr = args.lr
    epochs = args.epoch 
    if args.use_global:
        factor = args.group_number + 1
    elif args.use_group:
        factor = args.group_number
    else:
        factor = 1

    print("backbone lr", lr / factor)
    optimizer = torch.optim.SGD([
        {"params":model.backbone.parameters(),"lr":lr,"initial_lr":lr},
        {"params":model.group_heads.parameters(),"lr":lr ,"initial_lr":lr} ],
        weight_decay=1e-4, momentum=0.9)
    
    warmUp_scheduler = WarmUpLR(optimizer,total_iters=500)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*5/8), int(epochs*7.5/8)], gamma=0.1)
    
    print(args.loss_type)
    
    bce_criterion = BCEWithLogitsLoss()
    if args.loss_type == "bce":
        criterion_logit = BCEWithLogitsLoss()
    elif args.loss_type == "focal":
        criterion_logit = FocalLoss(balance_param=2,gamma=2)
    elif args.loss_type == "db":
        criterion_logit = ResampleLoss(freq_file=class_freq_file)
    elif args.loss_type == "asl":
        criterion_logit = ASL(disable_torch_grad_focal_loss=True,gamma_neg=args.gamma_neg,gamma_pos=args.gamma_pos)
    elif args.loss_type == "fb":
        criterion_logit = FBLoss(use_weight=False, use_focal=True,
                            gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos,num_classes=args.num_classes)
    else:
        exit(-1)

    
    highest_mAP = 0.0
    losses = AverageMeter()
    current_iters = 0
    for epoch in range(epochs):
        model.train()
        for i, (inputData, target) in enumerate(train_loader):
            # warm up lr 
            if current_iters < args.warm_iters:
                warmUp_scheduler.step()
                current_iters += 1

            inputData = inputData.cuda()
            target = target.float().cuda()
            
            if args.use_global:
                logits = model(inputData)[0]
                global_loss = bce_criterion(logits[0],target)
                group_loss = criterion_logit(logits[1],target)
                loss = (global_loss + group_loss) / 2            
            else:
                logits = model(inputData)[-1]
                loss = criterion_logit(logits, target)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.detach().cpu().numpy() * 10000,inputData.shape[0])
            
            writer.add_scalar("train_loss",losses.ema,len(train_loader) * epoch + i)

            if i % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.2f}'
                      .format(epoch + 1, epochs, str(i).zfill(3), str(len(train_loader)).zfill(3),
                              optimizer.param_groups[-1]['lr'],
                              losses.ema))
        
       # after warm up, use mileStone scheduler
        if current_iters >= args.warm_iters:
            scheduler.step()
        else:
            scheduler.last_epoch += 1

        print("Epoch [{}/{}] ".format(epoch+1, epochs))
        highest_mAP = validate(SPLIT_GROUP, used_groups, model, val_loader, epochs, epoch, highest_mAP)


def validate(SPLIT_GROUP, used_groups, model, val_loader, epochs, epoch,highest_mAP):
    model.eval()
    preds = []
    preds_group = []
    preds_global = []
    targets = []
    Sig = torch.nn.Sigmoid()
    for _ , (inputData, target) in enumerate(val_loader):
        inputData = inputData.cuda()            
        targets.append(target)

        with torch.no_grad():
            output = model(inputData)
            if args.use_global:
                out1 = Sig(output[0]).cpu()
                out2 = Sig(output[1]).cpu()
                preds_global.append(out1)
                preds_group.append(out2)

                # 更侧重global的head 和 分支模型的 local
                output_regular = (out1 + out2) / 2.0
                output_regular[:,used_groups[0]] = (0.8 * out1 + 0.2 * out2)[:,used_groups[0]] 
                output_regular[:,used_groups[-1]] = (0.2 * out1 + 0.8 * out2)[:,used_groups[-1]]
            else:
                output_regular = Sig(output[-1]).cpu()
                preds_group.append(output_regular)
        preds.append(output_regular)

    if args.use_group:
        head_score, middle_score, tail_score, mAP_score = computemAP(SPLIT_GROUP, preds_group, targets)
        print("Group Branch :  head {:.2f}, middle {:.2f}, tail {:.2f}, \
        total {:.2f}".format(head_score,
            middle_score,
            tail_score,
            mAP_score))

    if args.use_global:
        head_score, middle_score, tail_score, mAP_score = computemAP(SPLIT_GROUP, preds_global, targets)
        print("Global Branch :  head {:.2f}, middle {:.2f}, tail {:.2f}, \
        total {:.2f}".format(head_score,
            middle_score,
            tail_score,
            mAP_score))

    head_score, middle_score, tail_score, mAP_score = computemAP(SPLIT_GROUP, preds, targets)

    print("All Branch : head {:.2f}, middle {:.2f}, tail {:.2f}, \
        total {:.2f}".format(head_score,
            middle_score,
            tail_score,
            mAP_score))

    if mAP_score > highest_mAP:
        highest_mAP = mAP_score
        try:
            torch.save(model.state_dict(), os.path.join(
                    './checkpoint/', 'model-highest.ckpt'))
        except:
            pass

    writer.add_scalar("Test/mAP",mAP_score,epoch+1)
    writer.add_scalar("Test/Head",head_score,epoch+1)
    writer.add_scalar("Test/Middle",middle_score,epoch+1)
    writer.add_scalar("Test/Tail",tail_score,epoch+1)

    print("Epoch [{}/{}] head {:.2f}, middle {:.2f}, tail {:.2f}, \
        total {:.2f}, highest {:.2f}".format(epoch+1, epochs,head_score,
            middle_score,
            tail_score,
            mAP_score, highest_mAP))

    return highest_mAP

def computemAP(SPLIT_GROUP, preds, targets):
    targets = torch.cat(targets, dim=0).detach()
    preds = torch.cat(preds, dim=0).detach()
    _, aps = mAP(targets.cpu().numpy(), preds.cpu().numpy())

    head_score = np.mean(aps[SPLIT_GROUP[0]]) * 100
    middle_score = np.mean(aps[SPLIT_GROUP[1]]) * 100
    tail_score = np.mean(aps[SPLIT_GROUP[2]]) * 100
    mAP_score = np.mean(aps) * 100
    return head_score,middle_score,tail_score,mAP_score

if __name__ == '__main__':
    seed_torch(0)
    torch.cuda.set_device(args.device)
    main()

    # metadata = mmcv.load("./appendix/coco/longtail2017/metadata.pkl")
    # dict_keys(['gt_labels', 'class_freq', 'neg_class_freq', 'condition_prob', 'img_ids', 'cls_data_list', 'class_split'])
    