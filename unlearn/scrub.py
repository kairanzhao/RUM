import copy
import numpy as np
import wandb
import sys
import time
import torch
import torch.nn as nn
from itertools import cycle
import utils
from utils import DistillKL, AverageMeter, accuracy
from .impl import iterative_unlearn


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, max_iter, gamma, beta, split,
                  print_freq=12, quiet=False):
    """One epoch distillation"""
    # set modules as train()
    # for module in module_list:
    #     module.train()
    # # set teacher as eval()
    # module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]
    model_s.train()
    model_t.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    kd_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    acc_max_top1 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):

        input = input.cuda()
        target = target.cuda()
        data_time.update(time.time() - end)

        input = torch.Tensor(input).float()
        # target = torch.squeeze(torch.Tensor(target).long())

        # ===================forward=====================
        logit_s = model_s(input)
        with torch.no_grad():
            logit_t = model_t(input)

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        if split == "minimize":
            loss = gamma * loss_cls + beta * loss_div
        elif split == "maximize":
            loss = -loss_div

        if split == "minimize" and not quiet:
            acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))
        elif split == "maximize" and not quiet:
            kd_losses.update(loss.item(), input.size(0))
            acc_max, _ = accuracy(logit_s, target, topk=(1, 5))
            acc_max_top1.update(acc_max.item(), input.size(0))


    # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

    if split == "maximize":
        if not quiet:
            # if idx % print_freq == 0:
            print('*** Maximize step ***')
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Forget_Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=kd_losses, top1=acc_max_top1))
            # sys.stdout.flush()
    elif split == "minimize":
        if not quiet:
            print('*** Minimize step ***')
            # print(' * Acc@1 {top1.avg:.3f} '.format(top1=top1))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Retain_Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

        return top1.avg, losses.avg
    else:
        # module_list[0] = model_s
        # module_list[-1] = model_t
        return kd_losses.avg

@iterative_unlearn
def scrub(data_loaders, module_list, criterion, optimizer, epoch, args, mask=None):
    # model_ng = freeze_params(model_ng, args)
    f_loader = data_loaders["forget"]
    r_loader = data_loaders["retain"]
    print(f'len(r_loader): {len(r_loader)}, len(f_loader): {len(f_loader)}')

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    model_s = module_list[0]
    model_t = module_list[-1]
    start = time.time()

    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)
    criterion_kd = DistillKL(args.kd_T)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)
    criterion_list.append(criterion_div)
    criterion_list.append(criterion_kd)

    # criterion = torch.nn.CrossEntropyLoss().to(args.device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # adjust_learning_rate(epoch, lr_dict, optimizer)
    maximize_loss = 0
    iter_per_epoch_forget = 1
    iter_per_epoch_train = 1
    if epoch <= args.msteps:
        maximize_loss = train_distill(epoch, f_loader, module_list, criterion_list, optimizer,
                                      iter_per_epoch_forget, args.gamma, args.beta, "maximize", quiet=False)
    train_acc, train_loss = train_distill(epoch, r_loader, module_list, criterion_list, optimizer,
                                          iter_per_epoch_train, args.gamma, args.beta,"minimize", quiet=False)

    print(f"Epoch: [{epoch}]\t train-acc:\t{train_acc}\t train-loss: {train_loss}")


    return train_acc
