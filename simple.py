import os
import argparse
from random import sample
import torch
from src.models.model import NormModel,BaseModel
from datasets.coco import CocoDetection, LTCOCO
from src.helper_functions.helper_functions import mAP, CocoDetection, AverageMeter, WarmUpLR,seed_torch
import torchvision.transforms as transforms
import mmcv
from torch.nn import BCEWithLogitsLoss
from src.losses import AsymmetricLossOptimized
from losses import ResampleLoss,FocalLoss
import numpy as np
from torch.optim import lr_scheduler
from datasets.getDataset import getCOCODataset, getVocDataset
from src.models.pfc import PFC
from src.models.cls_head import ClsHead
from torch.utils.tensorboard import SummaryWriter
from src.sampler import m2sSample,ClassAwareSampler
from torchvision.models import resnet50
import torch.nn as nn

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=224, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--epoch', default=80, type=int, required=False)
parser.add_argument('--dataset-name',default="coco",type=str)
parser.add_argument('--use-m2s',action="store_true")
parser.add_argument('--device',type=int,default=0)
parser.add_argument('--use-rs',action="store_true")
parser.add_argument('--loss-type',type=str,default="bce")
parser.add_argument('--reduce',type=int,default=4)
parser.add_argument('--warm-iters',type=int,default=60)
parser.add_argument('--warm-radio',type=float,default=1.0 /3)
parser.add_argument('--output',type=str)

args = parser.parse_args()
#summary writer
args.output = os.path.join("/data/pengpeng/checkpoint/",args.output)
os.makedirs(args.output, exist_ok=True)
writer = SummaryWriter(log_dir=args.output)

save_point = []

def main():
    if args.dataset_name == "coco":
        train_dataset, val_dataset = getCOCODataset(use_m2s=args.use_m2s,image_size=args.image_size)
        # weights_save_path="/home/pengpeng/ASL/appendix/coco/lt2017/sample_weights.npy"
        class_freq_file="/home/pengpeng/ASL/appendix/coco/longtail2017/class_freq.pkl"
        SPLIT_GROUP = [[0, 2, 24, 26, 39, 41, 42, 43, 44, 45, 56, 57, 58, 60, 62, 63, 67, 69, 71, 72, 73, 75],
               [1, 3, 5, 7, 8, 9, 13, 15, 16, 25, 27, 28, 32, 34, 35, 38, 40, 46, 47, 48, 49, 50, 51,
                53, 55, 59, 61, 64, 65, 66, 68, 74, 77],
               [4, 6, 10, 11, 12, 14, 17, 18, 19, 20, 21, 22, 23, 29, 30, 31, 33, 36, 37, 52, 54, 70, 76, 78, 79]]
        num_classes = 80
        metadata = mmcv.load("/home/pengpeng/MLC/appendix/coco/longtail2017/metadata.pkl")

    elif args.dataset_name == "voc":
        train_dataset, val_dataset = getVocDataset(use_m2s=args.use_m2s,image_size=args.image_size)
        # weights_save_path = "/home/pengpeng/MLC/appendix/VOCdevkit/longtail2012/sample_weights.npy"
        class_freq_file = "/home/pengpeng/MLC/appendix/VOCdevkit/longtail2012/class_freq.pkl"
        SPLIT_GROUP = [[4,6,8,10,14,15],
            [1,7,11,13,17,19],
            [0,2,3,5,9,12,16,18]]
        num_classes = 20
        metadata = mmcv.load("/home/pengpeng/MLC/appendix/VOCdevkit/longtail2012/class_freq.pkl")
    else:
        print("unknown dataset name:", args.dataset_name)
        exit(-1)

    if args.use_rs:
        shuffle = False
        sampler = ClassAwareSampler(train_dataset,metadata,reduce=args.reduce)
    else:
        sampler = None
        shuffle = True    

    model = BaseModel(num_classes=num_classes)
    model = model.cuda()
    print("model finished!")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=shuffle,
        sampler = sampler,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size * 4, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    lr = args.lr
    epochs = args.epoch

    print(args.loss_type)
    if args.loss_type == "bce":
        criterion_logit = BCEWithLogitsLoss()
    elif args.loss_type == "asl":
        criterion_logit = AsymmetricLossOptimized(disable_torch_grad_focal_loss=False,gamma_neg=6,gamma_pos=0)
    elif args.loss_type == "focal":
        criterion_logit = FocalLoss(balance_param=2,gamma=2)
    elif args.loss_type == "db":
        criterion_logit = ResampleLoss(freq_file=class_freq_file)
    else:
        criterion_logit =None


    model_parameters = [{"params":model.parameters(),"initial_lr":lr, "lr":lr}]
    optimizer = torch.optim.SGD(
        model_parameters, lr=lr, weight_decay=1e-4, momentum=0.9)
    warmUp_scheduler = WarmUpLR(optimizer,total_iters=args.warm_iters)
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[5,7], gamma=0.1)

    highest_mAP = 0.0
    highest_aps = None
    losses = AverageMeter()
    current_iters = 0
    for epoch in range(epochs):
        model.train()

        preds = []
        targets = []
        Sig = torch.nn.Sigmoid()
        for i, (inputData, target) in enumerate(train_loader):
            # warm up lr 
            if current_iters < args.warm_iters:
                warmUp_scheduler.step()
                current_iters += 1

            inputData = inputData.cuda()
            targets.append(target.float())
            target = target.float().cuda()

            logits = model(inputData)
            loss = criterion_logit(logits, target)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.detach().cpu().numpy()*10000, inputData.shape[0])
            output_regular = Sig(logits.detach()).cpu()
            preds.append(output_regular)

            writer.add_scalar("train_loss",losses.ema,len(train_loader) * epoch + i)

            if i % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.2f}'
                      .format(epoch + 1, epochs, str(i).zfill(3), str(len(train_loader)).zfill(3),
                              optimizer.param_groups[0]['lr'],
                              losses.ema))

        # after warm up, use mileStone scheduler
        if current_iters >= args.warm_iters:
            scheduler.step()
        else:
            scheduler.last_epoch += 1

        targets = torch.cat(targets, dim=0).detach()
        preds = torch.cat(preds, dim=0).detach()
        _, aps = mAP(targets.cpu().numpy(), preds.cpu().numpy())

        head_score = np.mean(aps[SPLIT_GROUP[0]]) * 100
        middle_score = np.mean(aps[SPLIT_GROUP[1]]) * 100
        tail_score = np.mean(aps[SPLIT_GROUP[2]]) * 100
        mAP_score = np.mean(aps) * 100

        writer.add_scalar("Train/mAP",mAP_score,epoch+1)
        writer.add_scalar("Train/Head",head_score,epoch+1)
        writer.add_scalar("Train/Middle",middle_score,epoch+1)
        writer.add_scalar("Train/Tail",tail_score,epoch+1)

        model.eval()
        preds = []
        targets = []
        for i, (inputData, target) in enumerate(val_loader):
            inputData = inputData.cuda()
            
            targets.append(target)
            with torch.no_grad():
                output = model(inputData)
            output_regular = Sig(output).cpu()
            preds.append(output_regular)

        targets = torch.cat(targets, dim=0).detach()
        preds = torch.cat(preds, dim=0).detach()
        _, aps = mAP(targets.cpu().numpy(), preds.cpu().numpy())

        head_score = np.mean(aps[SPLIT_GROUP[0]]) * 100
        middle_score = np.mean(aps[SPLIT_GROUP[1]]) * 100
        tail_score = np.mean(aps[SPLIT_GROUP[2]]) * 100

        mAP_score = np.mean(aps) * 100

        if mAP_score > highest_mAP:
            highest_mAP = mAP_score
            highest_aps = aps
            try:
                torch.save(model.state_dict(), os.path.join(
                    args.output, 'model-highest.ckpt'))
            except:
                pass

        # if (epoch+1) % 10 == 0 and epoch != 0:
        #     torch.save(model.state_dict(), os.path.join(
        #         'models/', 'model-{}.ckpt'.format(epoch+1)))

        if (epoch + 1) in save_point:
            torch.save(model.state_dict(), os.path.join(
                args.output, 'model-{}.ckpt'.format(epoch+1)))

        writer.add_scalar("Test/mAP",mAP_score,epoch+1)
        writer.add_scalar("Test/Head",head_score,epoch+1)
        writer.add_scalar("Test/Middle",middle_score,epoch+1)
        writer.add_scalar("Test/Tail",tail_score,epoch+1)

        print("Epoch [{}/{}] head {:.2f}, middle {:.2f}, tail {:.2f}, total {:.2f}, highest {:.2f}".format(epoch+1, epochs,
                                                                                                           head_score,
                                                                                                           middle_score,
                                                                                                           tail_score,
                                                                                                           mAP_score, highest_mAP))
    print("aps \n",highest_aps)


if __name__ == '__main__':
    seed_torch(1037)
    torch.cuda.set_device(args.device)
    main()

    # metadata = mmcv.load("./appendix/coco/longtail2017/metadata.pkl")
    # dict_keys(['gt_labels', 'class_freq', 'neg_class_freq', 'condition_prob', 'img_ids', 'cls_data_list', 'class_split'])
