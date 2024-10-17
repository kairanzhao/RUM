import argparse
import os
import pdb
import pickle
import random
import shutil
import time
from copy import deepcopy
from rich import print as rich_print

import arg_parser
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from pruner import *
from torch.utils.data.sampler import SubsetRandomSampler
from trainer import train, validate
from utils import *
from utils import NormalizeByChannelMeanStd

from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

best_sa = 0


def wandb_init(args):
    if args.wandb_group_name is None:
        args.wandb_group_name = f"{args.dataset}_{args.arch}_{args.forget_class}_{args.num_to_forget}"
    if args.wandb_run_id is not None:
        logger = wandb.init(id=args.wandb_run_id, resume="must")
    else:
        run_name = "{}_{}_forget_{}_num{}_seed{}".format(args.dataset, args.arch,
                        args.class_to_replace, args.num_indexes_to_replace, args.seed)
        logger = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   mode=args.wandb_mode, group=args.wandb_group_name)
        logger.name = run_name

    logger.config.update(args)
    return logger
def wandb_finish(logger, files=None):
    if files is not None:
        for file in files:
            #if using wandb, save the latest model
            if isinstance(logger, type(wandb.run)) and logger is not None:
                #artifact = wandb.Artifact('model', type='model')
                #artifact.add_file(model_name)
                #logger.log_artifact(artifact)
                shutil.copyfile(file, os.path.join(logger.dir, os.path.basename(file)))
                #logger.save(file, policy='end')

    logger.finish()

def main():
    global args, best_sa
    args = arg_parser.parse_args()
    rich_print(args)

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        setup_seed(args.seed)

    logger = wandb_init(args)
    files_to_save = []

    # prepare dataset
    if args.dataset == "imagenet":
        args.class_to_replace = None
        model, train_loader, val_loader = setup_model_dataset(args)
    elif args.dataset == "cifar10" or args.dataset == "cifar100":
        (
            model,
            train_loader,
            val_loader,
            test_loader,
            marked_loader,
            train_idx
        ) = setup_model_dataset(args)
    elif args.dataset == "TinyImagenet":
        args.data_dir = "/data/image_data/tiny-imagenet-200/"
        (
            model,
            train_loader,
            val_loader,
            test_loader,
            marked_loader,
        ) = setup_model_dataset(args)
    model.cuda()

    print(f"number of train dataset {len(train_loader.dataset)}")
    print(f"number of val dataset {len(val_loader.dataset)}")

    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))

    """
    if args.prune_type == 'lt':
        print('lottery tickets setting (rewind to the same random init)')
        initalization = deepcopy(model.state_dict())
    elif args.prune_type == 'pt':
        print('lottery tickets from best dense weight')
        initalization = None
    elif args.prune_type == 'rewind_lt':
        print('lottery tickets with early weight rewinding')
        initalization = None
    else:
        raise ValueError('unknown prune_type')
    """

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    if args.imagenet_arch:
        lambda0 = (
            lambda cur_iter: (cur_iter + 1) / args.warmup
            if cur_iter < args.warmup
            else (
                0.5
                * (
                    1.0
                    + np.cos(
                        np.pi * ((cur_iter - args.warmup) / (args.epochs - args.warmup))
                    )
                )
            )
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
    else:
        # for training original model: cifar100-resnet50
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=decreasing_lr, gamma=0.2
        # )
        # for training original model: cifar10-resnet18
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.resume:
        print("resume from checkpoint {}".format(args.checkpoint))
        checkpoint = torch.load(
            args.checkpoint, map_location=torch.device("cuda:" + str(args.gpu))
        )
        best_sa = checkpoint["best_sa"]
        print(best_sa)
        start_epoch = checkpoint["epoch"]
        all_result = checkpoint["result"]

        """
        start_state = checkpoint['state']
        print(start_state)
        if start_state > 0:
            current_mask = extract_mask(checkpoint['state_dict'])
            prune_model_custom(model, current_mask)
            check_sparsity(model)
            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=decreasing_lr, gamma=0.1)
        """

        model.load_state_dict(checkpoint["state_dict"], strict=False)

        """
        # adding an extra forward process to enable the masks
        x_rand = torch.rand(1, 3, args.input_size, args.input_size).cuda()
        model.eval()
        with torch.no_grad():
            model(x_rand)
        """

        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        initalization = checkpoint["init_weight"]
        # print('loading state:', start_state)
        print("loading from epoch: ", start_epoch, "best_sa=", best_sa)

    else:
        all_result = {}
        all_result["train_ta"] = []
        all_result["test_ta"] = []
        all_result["val_ta"] = []

        start_epoch = 0
        state = 0
        # start_state = 0

    """
    print('######################################## Start Standard Training Iterative Pruning ########################################')

    for state in range(start_state, args.pruning_times):

        print('******************************************')
        print('pruning state', state)
        print('******************************************')

        check_sparsity(model)
    """

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        print(
            "Epoch #{}, Learning rate: {}".format(
                epoch, optimizer.state_dict()["param_groups"][0]["lr"]
            )
        )
        logger.log({'lr': optimizer.state_dict()["param_groups"][0]["lr"]})
        acc = train(train_loader, model, criterion, optimizer, epoch, args)
        logger.log({"train_acc": acc})

        """
        if state == 0:
            if (epoch+1) == args.rewind_epoch:
                torch.save(model.state_dict(), os.path.join(
                    args.save_dir, 'epoch_{}_rewind_weight.pt'.format(epoch+1)))
                if args.prune_type == 'rewind_lt':
                    initalization = deepcopy(model.state_dict())
        """

        # evaluate on validation set
        tacc = validate(val_loader, model, criterion, args)
        logger.log({"val_acc": tacc})
        # # evaluate on test set
        # test_tacc = validate(test_loader, model, criterion, args)

        scheduler.step()

        all_result["train_ta"].append(acc)
        all_result["val_ta"].append(tacc)
        # all_result['test_ta'].append(test_tacc)

        # remember best prec@1 and save checkpoint
        is_best_sa = tacc > best_sa
        best_sa = max(tacc, best_sa)

        """
        save_checkpoint({
            # 'state': state,
            'result': all_result,
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_sa': best_sa,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            # 'init_weight': initalization
        }, is_SA_best=is_best_sa, pruning=state, save_path=args.save_dir)
        """

        save_checkpoint(
            {
                "result": all_result,
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_sa": best_sa,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_SA_best=is_best_sa,
            pruning=state,
            save_path=args.save_dir,
        )
        print("one epoch duration:{}".format(time.time() - start_time))

    # plot training curve
    plt.plot(all_result["train_ta"], label="train_acc")
    plt.plot(all_result["val_ta"], label="val_acc")
    # plt.plot(all_result['test_ta'], label='test_acc')
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, str(state) + "net_train.png"))
    plt.close()

    model_name = os.path.join(args.save_dir, str(state) + "model_SA_best.pth.tar")
    save_checkpoint(
        {
            "state_dict": model.state_dict(),
            "best_sa": best_sa,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        is_SA_best=is_best_sa,
        pruning=state,
        save_path=args.save_dir,
        filename='{}_original_{}_bs{}_lr{}_seed{}_epochs{}.pth.tar'.format(
        args.dataset,args.arch,args.batch_size, args.lr, args.seed, args.epochs),
    )

    # report result
    # check_sparsity(model)
    print("Performance on the val data set")
    test_tacc = validate(val_loader, model, criterion, args)
    if len(all_result["val_ta"]) != 0:
        val_pick_best_epoch = np.argmax(np.array(all_result["val_ta"]))
        print(
            "* best SA = {}, Epoch = {}".format(
                all_result["val_ta"][val_pick_best_epoch], val_pick_best_epoch + 1
            )
        )

        """
        all_result = {}
        all_result['train_ta'] = []
        all_result['test_ta'] = []
        all_result['val_ta'] = []
        best_sa = 0
        start_epoch = 0

        if args.prune_type == 'pt':
            print('* loading pretrained weight')
            initalization = torch.load(os.path.join(
                args.save_dir, '0model_SA_best.pth.tar'), map_location=torch.device('cuda:'+str(args.gpu)))['state_dict']

        #pruning and rewind
        if args.random_prune:
            print('random pruning')
            pruning_model_random(model, args.rate)
        else:
            print('L1 pruning')
            pruning_model(model, args.rate)

        remain_weight = check_sparsity(model)
        current_mask = extract_mask(model.state_dict())
        remove_prune(model)

        # weight rewinding
        # rewind, initialization is a full model architecture without masks
        model.load_state_dict(initalization, strict=False)
        prune_model_custom(model, current_mask)
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        if args.imagenet_arch:    
            lambda0 = lambda cur_iter: (cur_iter+1) / args.warmup if cur_iter < args.warmup else \
                (0.5*(1.0+np.cos(np.pi*((cur_iter-args.warmup)/(args.epochs-args.warmup)))))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda0)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=decreasing_lr, gamma=0.1)  # 0.1 is fixed
        if args.rewind_epoch:
            # learning rate rewinding
            for _ in range(args.rewind_epoch):
                scheduler.step()
        """


if __name__ == "__main__":
    main()
