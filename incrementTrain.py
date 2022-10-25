import os
import argparse
import torch
from torch.optim import lr_scheduler
from src.models.model import MultiStageModel, FreezeModel
from src.dataset import FlexibleDataset
from src.helper_functions.helper_functions import mAP, CocoDetection, voc_mAP
import torchvision.transforms as transforms
import mmcv
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, MSELoss
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--data', metavar='DIR', help='path to dataset', default='/home/MSCOCO_2014/')
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--model-path', default=None, type=str)
parser.add_argument('--num-classes', default=80)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=224, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--thre', default=0.8, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument("--reverse", action="store_true")
parser.add_argument("--epoch", type=int, default=40, required=False)


def main():
    args = parser.parse_args()
    metadata = mmcv.load("/home/pengpeng/MLC/appendix/coco/longtail2017/metadata.pkl")
    groups = metadata["class_split"]
    print("using reverse")
    if args.reverse:
        groups.reverse()
    model = MultiStageModel(groups=groups)
    model = model.cuda()
    print("model finished!")

    args.data = "/data/coco/"
    data_path_train = f'{args.data}/train2017'  # args.data
    data_path_val = f'{args.data}/val2017'
    instances_path_val = os.path.join(args.data, 'annotations/instances_val2017.json')
    instances_path_train = os.path.join(args.data, 'annotations/instances_train2017.json')
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    # group_ids 包含每个阶段使用的图片id
    save_path = '/home/pengpeng/ASL/appendix/coco/IncLearning/group_ids.pkl'
    group_ids = mmcv.load(save_path)
    # 如果需要反向训练
    if args.reverse:
        group_ids.reverse()

    num_stages = len(group_ids)

    # 准备train dataset 和 test dataset
    train_loaders = []
    for stage in range(num_stages):
        train_dataset = FlexibleDataset(data_path_train,
                                        instances_path_train,
                                        group_ids[stage],
                                        transforms.Compose([
                                            transforms.Resize((args.image_size, args.image_size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            normalization
                                        ]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
        train_loaders.append(train_loader)

    val_dataset = CocoDetection(data_path_val,
                                instances_path_val,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    normalization
                                ]))

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size * 4, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    epochs = [args.epoch, args.epoch, args.epoch]
    lr = args.lr

    criterion_logit = BCEWithLogitsLoss()
    criterion_feas = MSELoss()

    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9)
    scheduler = None

    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    old_model = FreezeModel(model)

    for stage in range(len(groups)):
        # 对于第一阶段后的所有阶段 需要使用蒸馏学习模式，保存之前学习阶段的知识
        if stage > 0:
            print("set last checkpoint!")
            old_model.set(model)

        train_loader = train_loaders[stage]

        for epoch in range(epochs[stage]):
            model.train()
            model.set_stage(stage)
            for i, (inputData, target) in enumerate(train_loader):
                inputData = inputData.cuda()
                # print(target.shape) # [batch_size, num_classes]
                target = target[:, groups[stage]].float().cuda()
                logit, feas = model(inputData)
                new_fea = feas["1"]
                # 计算蒸馏损失
                logit_loss = criterion_logit(logit, target)
                if stage > 0:
                    with torch.no_grad():
                        _, old_fea = old_model.module(inputData)
                        old_fea = old_fea["1"].detach()
                    dis_loss = criterion_feas(new_fea, old_fea)
                else:
                    dis_loss = 0 * logit_loss
                loss = logit_loss + dis_loss

                model.zero_grad()
                loss.backward()
                optimizer.step()

                if i % 20 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.2f}, Logit: {:.2f}, Fea: {:.2f}'
                          .format(epoch + 1, epochs, str(i).zfill(3), str(len(train_loader)).zfill(3),
                                  optimizer.param_groups[0]['lr'],
                                  loss.item(), logit_loss.item(), dis_loss.item()))

            if scheduler is not None:
                scheduler.step()

            model.eval()

            preds = []
            targets = []
            Sig = torch.nn.Sigmoid()
            new_group = []
            # new group 是 logit输出的 对应的 类别 映射
            for i in range(len(groups)):
                new_group.extend(groups[i])

            for i, (inputData, target) in enumerate(val_loader):
                inputData = inputData.cuda()
                target = target.max(dim=1)[0].float()
                # 映射一下
                targets.append(target[:, new_group])
                with torch.no_grad():
                    outputs, _ = model(inputData)
                # 使用cat 的原因 是这里 outputs 输出的是 好几个组的结果，使用cat 将其连接起来
                output = torch.cat(outputs, dim=1)
                output_regular = Sig(output).cpu()
                preds.append(output_regular)

            targets = torch.cat(targets, dim=0).detach()
            preds = torch.cat(preds, dim=0).detach()
            score, aps = mAP(targets.cpu().numpy(), preds.cpu().numpy())

            head_score = np.mean(aps[:len(groups[0])]) * 100
            middle_score = np.mean(aps[len(groups[0]):len(groups[0]) + len(groups[1])]) * 100
            tail_score = np.mean(aps[len(groups[0]) + len(groups[1]):]) * 100

            end = 0
            for i in range(stage + 1):
                end = end + len(groups[i])
            total_score = np.mean(aps[:end]) * 100
            print("head {:.2f}, middle {:.2f}, tail {:.2f}, total {:.2f}".format(head_score, middle_score, tail_score,
                                                                                 total_score))


if __name__ == '__main__':
    torch.cuda.set_device(0)
    main()
