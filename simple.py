from lib2to3.pytree import Base
import os
import argparse
import torch
from src.models.model import NormModel
from datasets.coco import CocoDetection, LTCOCO
from src.helper_functions.helper_functions import mAP, CocoDetection
import torchvision.transforms as transforms
import mmcv
# from torch.nn import BCEWithLogitsLoss
# from src.losses import AsymmetricLoss as ASL
from src.losses import BCELoss as BCEWithLogitsLoss
import numpy as np
from torch.optim import lr_scheduler

from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--num-classes', default=80)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=224, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--epoch', default=80, type=int, required=False)

SPLIT_GROUP = [[0, 2, 24, 26, 39, 41, 42, 43, 44, 45, 56, 57, 58, 60, 62, 63, 67, 69, 71, 72, 73, 75],
               [1, 3, 5, 7, 8, 9, 13, 15, 16, 25, 27, 28, 32, 34, 35, 38, 40, 46, 47, 48, 49, 50, 51,
                53, 55, 59, 61, 64, 65, 66, 68, 74, 77],
               [4, 6, 10, 11, 12, 14, 17, 18, 19, 20, 21, 22, 23, 29, 30, 31, 33, 36, 37, 52, 54, 70, 76, 78, 79]]


writer = SummaryWriter(log_dir="./logs/"+"normModel")

def main():
    args = parser.parse_args()
    model = NormModel()
    model = model.cuda()
    print("model finished!")

    # metadata = mmcv.load("./appendix/coco/longtail2017/metadata.pkl")
    # root = "/home/share1/coco/"
    root = "/data/coco"

    data_path_train = f'{root}/train2017'
    data_path_val = f'{root}/val2017'

    instances_path_train = os.path.join(
        root, 'annotations/instances_train2017.json')
    instances_path_val = os.path.join(
        root, 'annotations/instances_val2017.json')

    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    train_dataset = LTCOCO(data_path_train, instances_path_train,
                           LT_ann_file="/home/pengpeng/MLC/appendix/coco/longtail2017/img_id.pkl",
                           transform=transforms.Compose([
                               transforms.Resize((224, 224)),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               normalization
                           ]))

    


    val_dataset = CocoDetection(data_path_val,
                                instances_path_val,
                                transforms.Compose([
                                    transforms.Resize(
                                        (args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    normalization
                                ]))

    print("len(val_dataset)): ", len(val_dataset))
    print("len(train_dataset)): ", len(train_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size * 4, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    lr = args.lr
    epochs = args.epoch
    criterion_logit = BCEWithLogitsLoss()
    # criterion_logit = ASL(disable_torch_grad_focal_loss=True,gamma_neg=0,gamma_pos=0)
    optimizer = torch.optim.SGD(
        params=model.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9)
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[55, 75], gamma=0.1)

    highest_mAP = 0.0
    for epoch in range(epochs):
        model.train()

        preds = []
        targets = []
        Sig = torch.nn.Sigmoid()
        for i, (inputData, target) in enumerate(train_loader):
            inputData = inputData.cuda()
            target = target.max(dim=1)[0].float()

            targets.append(target.float())
            target = target.float().cuda()

            logits = model(inputData)
            loss = criterion_logit(logits, target)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            output_regular = Sig(logits.detach()).cpu()
            preds.append(output_regular)

            if i % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.2f}'
                      .format(epoch + 1, epochs, str(i).zfill(3), str(len(train_loader)).zfill(3),
                              optimizer.param_groups[0]['lr'],
                              loss.item()))

        scheduler.step()

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
            target = target.max(dim=1)[0].float()
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

        # if mAP_score > highest_mAP:
        #     highest_mAP = mAP_score
        #     try:
        #         torch.save(model.state_dict(), os.path.join(
        #             'models/', 'model-highest.ckpt'))
        #     except:
        #         pass

        # if (epoch+1) % 10 == 0 and epoch != 0:
        #     torch.save(model.state_dict(), os.path.join(
        #         'models/', 'model-{}.ckpt'.format(epoch+1)))

        writer.add_scalar("Test/mAP",mAP_score,epoch+1)
        writer.add_scalar("Test/Head",head_score,epoch+1)
        writer.add_scalar("Test/Middle",middle_score,epoch+1)
        writer.add_scalar("Test/Tail",tail_score,epoch+1)

        print("Epoch [{}/{}] head {:.2f}, middle {:.2f}, tail {:.2f}, total {:.2f}, highest {:.2f}".format(epoch+1, epochs,
                                                                                                           head_score,
                                                                                                           middle_score,
                                                                                                           tail_score,
                                                                                                           mAP_score, highest_mAP))


if __name__ == '__main__':
    torch.cuda.set_device(2)
    main()

    # metadata = mmcv.load("./appendix/coco/longtail2017/metadata.pkl")
    # dict_keys(['gt_labels', 'class_freq', 'neg_class_freq', 'condition_prob', 'img_ids', 'cls_data_list', 'class_split'])
