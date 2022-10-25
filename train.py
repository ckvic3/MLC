from ast import For
import os
import argparse
import torch
from src.models.model import NormModel
from src.models.group import BaseModel
from src.helper_functions.helper_functions import mAP, CocoDetection
import torchvision.transforms as transforms
import mmcv
from src.losses import ClassWiseLoss, ReBalancedLoss
from src.dataset import SDataset, FlexibleDataset,CustomDataset
from torch.nn import BCEWithLogitsLoss
import numpy as np

from torch.utils.data import DataLoader
from src.sampler import ClassAwareSampler

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--data', metavar='DIR', help='path to dataset', default='/home/MSCOCO_2014/')
parser.add_argument('--model-name', type=str, default="base", required=False)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--model-path', default=None, type=str)
parser.add_argument('--num-classes', default=80)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=224, type=int,
                    metavar='N', help='input image size (default: 224)')
parser.add_argument('--thre', default=0.8, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--print-freq', '-p', default=64, type=int,
                    metavar='N', help='print frequency (default: 64)')
parser.add_argument('--use-resample', action="store_true")
parser.add_argument('--loss-type', default="bce", type=str, required=False)
parser.add_argument('--epoch', default=80, type=int, required=False)
parser.add_argument('--frozen-stage', default=1, type=int, required=False)

temp = mmcv.load("./appendix/coco/longtail2017/class_split.pkl")

groups = [list(temp["head"]), list(temp["middle"]), list(temp["tail"])]


def main():
    args = parser.parse_args()
    if args.model_name == "base":
        model = BaseModel(frozen_stages=args.frozen_stage)
    elif args.model_name == "norm":
        model = NormModel(frozen_stages=args.frozen_stage)
    else:
        raise NameError
    print("using {} model".format(args.model_name))
    model = model.cuda()
    print("model finished!")

    # metadata = mmcv.load("./appendix/coco/longtail2017/metadata.pkl")
    metadata = None
    root = "/home/share1/coco/"

    data_path_train = f'{root}/train2017'
    data_path_val = f'{root}/val2017'

    instances_path_train = os.path.join(root, 'annotations/instances_train2017.json')
    instances_path_val = os.path.join(root, 'annotations/instances_val2017.json')

    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    print("img size is :", args.image_size)

    multi_label_file = "/home/pengpeng/ASL/appendix/coco/longtail2017/img_ids.npy"
    train_dataset = CustomDataset(data_path_train,
                                  instances_path_train,
                                  multi_label_file,
                                  transforms.Compose([
                                      transforms.Resize((args.image_size, args.image_size)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      normalization
                                  ]),
                                  need_masked=False,
                                  )

    # train_dataset = FlexibleDataset(data_path_train,
    #                                 instances_path_train,
    #                                 metadata["img_ids"],
    #                                 transforms.Compose([
    #                                     transforms.Resize((args.image_size, args.image_size)),
    #                                     transforms.RandomHorizontalFlip(),
    #                                     transforms.ToTensor(),
    #                                     normalization
    #                                 ])
    #                                 )

    val_dataset = CocoDetection(data_path_val,
                                instances_path_val,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    normalization
                                ]))

    train_loader = None
    epochs = args.epoch
    if args.use_resample:
        print("using class aware resampling")
        sampler = ClassAwareSampler(train_dataset, metadata)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False,
            sampler=sampler, num_workers=args.workers, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
        epochs = epochs * 4

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size * 4, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    criterion_logit = None
    if args.loss_type == "bce":
        criterion_logit = BCEWithLogitsLoss()
    elif args.loss_type == "class_wise":
        criterion_logit = ClassWiseLoss()
    elif args.loss_type == "rbce":
        criterion_logit = ReBalancedLoss()
    print("using {} loss".format(args.loss_type))

    lr = args.lr

    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9)
    # milestones = [100]
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    weight_aps = None
    for epoch in range(epochs):
        if args.loss_type == "class_wise" and weight_aps is not None:
            criterion_logit.set_weight(weights=weight_aps)
        model.train()

        preds = []
        targets = []
        Sig = torch.nn.Sigmoid()
        for i, (inputData, target) in enumerate(train_loader):
            inputData = inputData.cuda()

            targets.append(target.float())

            target = target.float().cuda()
            logits = model(inputData)
            loss = criterion_logit(logits, target)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            output_regular = Sig(logits.detach()).cpu()
            preds.append(output_regular)

            if i % 20 == 0:
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.2f}'
                      .format(epoch + 1, epochs, str(i).zfill(3), str(len(train_loader)).zfill(3),
                              optimizer.param_groups[0]['lr'],
                              loss.item()))

        if args.loss_type == "class_wise":
            targets = torch.cat(targets, dim=0).detach()
            preds = torch.cat(preds, dim=0).detach()
            score, weight_aps = mAP(targets.cpu().numpy(), preds.cpu().numpy())

        # scheduler.step()

        model.eval()
        preds = []
        targets = []
        # Sig = torch.nn.Sigmoid()
        for i, (inputData, target) in enumerate(val_loader):
            inputData = inputData.cuda()
            target = target.max(dim=1)[0].float()
            targets.append(target)
            with torch.no_grad():
                output = model(inputData)
            output_regular = Sig(output).cpu()
            preds.append(output_regular)

        targets = torch.cat(targets, dim=0).detach()
        preds = torch.cat(preds, dim=0).detach()
        score, aps = mAP(targets.cpu().numpy(), preds.cpu().numpy())

        head_score = np.mean(aps[groups[0]]) * 100
        middle_score = np.mean(aps[groups[1]]) * 100
        tail_score = np.mean(aps[groups[2]]) * 100

        total_score = np.mean(aps) * 100
        print("head {:.2f}, middle {:.2f}, tail {:.2f}, total {:.2f}".format(head_score, middle_score, tail_score,
                                                                             total_score))


if __name__ == '__main__':
    torch.cuda.set_device(0)
    main()


