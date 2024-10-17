import time
from copy import deepcopy

import numpy as np
import torch
import utils
from torch.utils.data import Subset

from .impl import iterative_unlearn

from .impl import wandb_init, wandb_finish


@iterative_unlearn
def RL_og(data_loaders, model, criterion, optimizer, epoch, args, mask=None):

    logger = wandb_init(args)

    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    forget_dataset = deepcopy(forget_loader.dataset)
    if args.dataset == "cifar10":
        if args.group_index or args.mem is not None:
            original_dataset = forget_dataset.dataset
            original_targets = original_dataset.targets
            forget_targets = [original_targets[idx] for idx in forget_dataset.indices]
            forget_targets = np.random.randint(
                0, args.num_classes, len(forget_targets)
            )
            for idx, label in zip(forget_dataset.indices, forget_targets):
                original_dataset.targets[idx] = label
            forget_dataset = Subset(original_dataset, forget_dataset.indices)

        else:
            forget_dataset.targets = np.random.randint(
                0, args.num_classes, forget_dataset.targets.shape
            )
    elif args.dataset == "svhn":
        forget_dataset.labels = np.random.randint(
            0, args.num_classes, forget_dataset.labels.shape
        )

    retain_dataset = retain_loader.dataset
    train_dataset = torch.utils.data.ConcatDataset([forget_dataset, retain_dataset])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    loader_len = len(forget_loader) + len(retain_loader)

    if epoch < args.warmup:
        utils.warmup_lr(epoch, i + 1, optimizer, one_epoch_step=loader_len, args=args)


    for it, (image, target) in enumerate(train_loader):
        i = it + len(forget_loader)
        image = image.cuda()
        target = target.cuda()
        # compute output
        output_clean = model(image)
        loss = criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()

        if mask:
            # print("Applying mask for SalUn!")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name]
                    # print(mask[name])
        else:
            # print("No mask applied!")
            pass

        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = utils.accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if (i + 1) % args.print_freq == 0:
            end = time.time()
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                "Time {3:.2f}".format(
                    epoch, i, loader_len, end - start, loss=losses, top1=top1
                )
            )
            start = time.time()

    logger.log({"train_acc": top1.avg, "train_loss": losses.avg})

    return top1.avg
