import os
import argparse
import torch
from src.models import createModelFactory
from src.helper_functions.helper_functions import mAP
import mmcv
import numpy as np
from torch.optim import lr_scheduler
from src.helper_functions.helper_functions import seed_torch, WarmUpLR, AverageMeter
from mmcv import Config
import datetime
from dataset import getDataloader
from torch.utils.tensorboard import SummaryWriter
from utils import Logger
from losses import createLossFuntion
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--model-path', default=None, type=str)
parser.add_argument('--thre', default=0.8, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('--print-freq', '-p', default=64, type=int,
                    metavar='N', help='print frequency (default: 64)')
parser.add_argument('--device',default=9,type=int)
parser.add_argument('--config',type=str,default="./configs/coco/coco_base_bce.py")
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--seed',type=int,default=0)
args = parser.parse_args()

def main():
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(dict(args._get_kwargs()))
    
    now = datetime.datetime.now()
    output_path = os.path.join(cfg.output_path,f"{now.date()}-{now.hour}-{now.minute}".replace(' ', '-'))
    os.makedirs(output_path)
    
    os.system('cp {} {}'.format(args.config,output_path))
    log = Logger(filename=os.path.join(output_path,'log.txt'))
    writer = SummaryWriter(log_dir=output_path)
    
    model = createModelFactory(cfg)
    model = model.cuda()
    
    train_loader, val_loader = getDataloader(cfg)
    epochs = cfg.epochs

    criterion= createLossFuntion(cfg)

    optimizer = torch.optim.SGD(params=model.parameters(),weight_decay=1e-4,momentum=0.9,lr=cfg.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones, gamma=0.1)

    if cfg.warm_up['total_iters'] > 0:
        warmup_scheduler = WarmUpLR(optimizer,total_iters=cfg.warm_up['total_iters'])
    
    
    ema_loss = AverageMeter()
    cur_step = -1
    highest_mAP = 0.0
    for epoch in range(epochs):
        # if args.loss_type == "class_wise" and weight_aps is not None:
        #     criterion.set_weight(weights=weight_aps)
        model.train()

        for i, (imgs, targets, masks) in enumerate(train_loader):
            cur_step += 1
            if cur_step < cfg.warm_up['total_iters']:
                warmup_scheduler.step()
                
            # targets.append(target.float())
            targets = targets.cuda()
            imgs=imgs.cuda()
            
            output = model(imgs)
            if cfg.model['name'] != 'group':
                loss = criterion(output.float(), targets, masks)
            else:
                if cfg.model['param']['mode'] == 'local':
                    loss = criterion(output[0].float(), targets, masks)
                elif cfg.model['param']['mode'] == 'global':
                    loss = criterion(output[1].float(), targets, masks)
                elif cfg.model['param']['mode'] == 'fusion':
                    loss = criterion(output[0].float(), targets, masks) + criterion(output[1].float(), targets, masks)
                else:
                    raise NotImplementedError
            
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 数值放大，避免记录的值过小
            loss = loss * args.batch_size * cfg.model['param']['num_classes']
          
            # ema update loss
            ema_loss.update(loss.detach().cpu().numpy(),n=imgs.shape[0])

            writer.add_scalar('train/ema_loss',ema_loss.ema,cur_step)
            writer.add_scalar('train/loss',loss.detach().cpu().item(), cur_step)
            writer.add_scalar("lr", optimizer.param_groups[0]['lr'], cur_step)


            # output_regular = Sig(logits.detach()).cpu()
            # preds.append(output_regular)

            if i % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.2f}'
                      .format(epoch + 1, epochs, str(i).zfill(3), str(len(train_loader)).zfill(3),
                              optimizer.param_groups[0]['lr'],
                              loss.item()))

        # if args.loss_type == "class_wise":
        #     targets = torch.cat(targets, dim=0).detach()
        #     preds = torch.cat(preds, dim=0).detach()
        #     score, weight_aps = mAP(targets.cpu().numpy(), preds.cpu().numpy())
        if cur_step >= cfg.warm_up['total_iters']:
            scheduler.step()
    
        mAP, aps = validate(model,val_loader,cfg,log)

        head_score = aps[cfg.split_group[0]].mean()
        middle_score = aps[cfg.split_group[1]].mean()
        tail_score = aps[cfg.split_group[2]].mean()

        writer.add_scalar('val/head', head_score, epoch)
        writer.add_scalar('val/middle', middle_score, epoch)
        writer.add_scalar('val/tail', tail_score, epoch)
        writer.add_scalar("val/val mAP", mAP, epoch)

        if (epoch + 1) % 10 ==0:
            try:
                torch.save({"state": model.state_dict(), "mAP": mAP},
                           os.path.join(output_path, 'model-{}.ckpt'.format(epoch+1)))
            except:
                log.logger.info("save weight failed")

        if mAP > highest_mAP:
            highest_mAP = mAP
            try:
                torch.save({"state": model.state_dict(), "mAP": mAP},
                           os.path.join(output_path, 'model-highest.ckpt'))
            except:
                pass
        log.logger.info("head: {:.2f}, middle: {:.2f}, tail: {:.2f}".format(head_score, middle_score, tail_score))
        log.logger.info('current_mAP = {:.2f}, highest_mAP = {:.2f}'.format(mAP, highest_mAP))



def validate(model,loader,cfg,log):
    model.eval()
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    targs = []
    preds = []
    for _,(imgs,targets,_) in enumerate(loader):
        imgs = imgs.cuda()
    
        with torch.no_grad():
            output = model(imgs)

        if cfg.model['name'] != 'group':
            output_regular = Sig(output).cpu()
        else:
            if cfg.model['param']['mode'] == 'local':
                output_regular = Sig(output[0]).cpu()
            elif cfg.model['param']['mode'] == 'global':
                output_regular = Sig(output[1]).cpu()
            elif cfg.model['param']['mode'] == 'fusion':
                output_regular = (Sig(output[0]).cpu() + Sig(output[1]).cpu()) / 2.0
            else:
                raise NotImplementedError
        
        targs.append(targets)
        preds.append(output_regular.detach().cpu())

    model.train() 
    targs = torch.cat(targs,0).numpy()
    preds = torch.cat(preds,0).numpy()
 
    mean_ap, aps = mAP(targs, preds)  

    preds = (preds > 0.5).astype(int)
    f1 = f1_score(targs,preds,average=None)
    log.logger.info("f1 score: {:2f}".format(f1.mean()))
    return mean_ap, aps

if __name__ == '__main__':
    seed_torch(0)
    torch.cuda.set_device(args.device)
    main()
