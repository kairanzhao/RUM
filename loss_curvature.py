import os
import numpy as np
import yaml
from tqdm import tqdm
import argparse
import copy
import shutil
import time
import collections
from rich import print as rich_print
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.optim.lr_scheduler import CosineAnnealingLR
from trainer import train, validate
from trainer.val import validate_withids
import arg_parser
import utils
from utils import *
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

def load_config(file_path, exp_key):
    with open(file_path, "r") as file:
        all_configs = yaml.safe_load(file)
        configs = all_configs.get(exp_key)
        return argparse.Namespace(**configs)

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)


def find_z(net, inputs, targets, criterion, h, device):
    ''' Adopted with modification from SOURCE: https://github.com/F-Salehi/CURE_robustness/blob/master/CURE/CURE.py
    the criterion reduction is set to non to get the loss for each sample

    Finding the direction in the regularizer
    '''
    criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    inputs.requires_grad_()
    outputs = net.eval()(inputs)
    loss_z = criterion(net.eval()(inputs), targets)

    loss_z.backward(torch.ones(targets.size()).to(device))
    # loss_z.backward()
    grad = inputs.grad.data + 0.0
    norm_grad = grad.norm().item()
    z = torch.sign(grad).detach() + 0.
    z = 1. * (h) * (z + 1e-7) / (z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None] + 1e-7)
    zero_gradients(inputs)
    net.zero_grad()

    return z, norm_grad


def regularizer(net, inputs, targets, criterion, h, device):
    '''Adopted with modification from SOURCE: https://github.com/F-Salehi/CURE_robustness/blob/master/CURE/CURE.py
    the criterion reduction is set to non to get the loss for each sample
    also the grad_diff is not aggregated across the batch

    Regularizer term in CURE
    '''
    criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    z, norm_grad = find_z(net, inputs, targets, criterion, h, device)

    inputs.requires_grad_()
    outputs_pos = net.eval()(inputs + z)
    outputs_orig = net.eval()(inputs)

    loss_pos = criterion(outputs_pos, targets)
    loss_orig = criterion(outputs_orig, targets)
    # grad_diff = torch.autograd.grad((loss_pos - loss_orig), inputs, grad_outputs=torch.ones(targets.size()).to(device),
    #                                 create_graph=True)[0]
    grad_diff = torch.autograd.grad((loss_pos - loss_orig), inputs, grad_outputs=torch.ones(targets.size()).to(device),
                                    create_graph=False)[0]

    reg = grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1)
    net.zero_grad()

    # return torch.sum(reg) / float(inputs.size(0)), norm_grad
    return reg, norm_grad

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target, _ in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def main():
    global args, best_sa
    args = arg_parser.parse_args()
    rich_print(args)

    torch.cuda.set_device(int(args.gpu))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        setup_seed(args.seed)

    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = False

    args.h = [float(i) for i in args.h.split(',')]
    if len(args.h)>args.epochs:
        raise ValueError('Length of h should be less than number of epochs')
    if len(args.h)==1:
        args.h = args.epochs * [args.h[0]]
    else:
        h_all = args.epochs * [1.0]
        h_all[:len(args.h)] = list(args.h[:])
        h_all[len(args.h):] = (args.epochs - len(args.h)) * [args.h[-1]]
        args.h = h_all

    logger = wandb_init(args)
    files_to_save = []

    if args.dataset == "TinyImagenetwithids":
        args.data_dir = "/data/image_data/tiny-imagenet-200/"
        print(args.data_dir)

    # prepare dataset
    if args.dataset == "imagenet":
        args.class_to_replace = None
        model, train_loader, val_loader = setup_model_dataset(args)
    else:
        (
            model,
            train_loader,
            val_loader,
            test_loader,
            marked_loader,
            # train_idx
        ) = setup_model_dataset(args)
    model.cuda()

    print(f"number of train dataset {len(train_loader.dataset)}")
    print(f"number of val dataset {len(val_loader.dataset)}")

    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))

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
            else (0.5* (1.0+ np.cos(np.pi * ((cur_iter - args.warmup) / (args.epochs - args.warmup)))))
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
    else:
        if args.dataset == "cifar10withids" or args.dataset == "TinyImagenetwithids":
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        elif args.dataset == "cifar100withids":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=decreasing_lr, gamma=0.2
            )

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

        train_curvs = collections.defaultdict(list)
        total_confidence_time = 0

        for epoch in range(start_epoch, args.epochs):
            start = time.time()
            print("Epoch #{}, Learning rate: {}".format(
                    epoch, optimizer.state_dict()["param_groups"][0]["lr"]))
            logger.log({'lr': optimizer.state_dict()["param_groups"][0]["lr"]})

            running_loss = 0
            correct_predictions = 0
            total_predictions = 0

            losses = utils.AverageMeter()
            top1 = utils.AverageMeter()

            epoch_confidence_time = 0
            for i, (images, labels, unique_ids) in enumerate(train_loader):
                images, labels, unique_ids = images.to(args.device), labels.to(args.device), unique_ids.to(args.device)
                model.train()
                optimizer.zero_grad()
                outputs = model(images)

                start_con = time.time()

                curv, grad_norm = regularizer(model, images, labels, criterion, h=args.h[epoch], device=args.device)

                for _id, _c in zip(unique_ids, curv):
                    train_curvs[_id.item()].append(_c.item())

                end_con = time.time()
                epoch_confidence_time += (end_con - start_con)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                model.eval()
                # Calculate softmax probabilities and predictions
                softmax_probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

                prec1 = utils.accuracy(outputs, labels)[0]
                losses.update(loss.item(), images.size(0))
                top1.update(prec1.item(), images.size(0))
                if (i + 1) % args.print_freq == 0:
                    end = time.time()
                    print(
                        "Epoch: [{0}][{1}/{2}]\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                        "Time {3:.2f}".format(
                            epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                        )
                    )
                    start = time.time()

            print("train_accuracy {top1.avg:.3f}".format(top1=top1))
            print(f"Epoch {epoch} proxy time: {epoch_confidence_time:.3f}")
            total_confidence_time += epoch_confidence_time

            training_accuracy = 100 * correct_predictions / total_predictions
            logger.log({"train_acc": training_accuracy})
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {running_loss/len(train_loader):.4f}, Training Accuracy: {training_accuracy:.2f}%")

            # evaluate on validation set
            tacc = validate_withids(val_loader, model, criterion, args)
            logger.log({"val_acc": tacc})

            scheduler.step()

            acc = top1.avg
            all_result["train_ta"].append(acc)
            all_result["val_ta"].append(tacc)
            # all_result['test_ta'].append(test_tacc)

            # remember best prec@1 and save checkpoint
            is_best_sa = tacc > best_sa
            best_sa = max(tacc, best_sa)


        # save the final model
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
            filename='{}_original_{}_bs{}_lr{}_seed{}_epochs{}_loss_curvature.pth.tar'.format(
            args.dataset,args.arch,args.batch_size, args.lr, args.seed, args.epochs),
        )

        average_confidence_time = total_confidence_time / args.epochs
        print(f"Total proxy time cost: {total_confidence_time:.3f} seconds")
        print(f"Avg proxy time cost per epoch: {average_confidence_time:.3f} seconds")

        # report result
        print("Performance on the test data set")
        test_tacc = validate_withids(val_loader, model, criterion, args)
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


        # Convert train_curvs dictionary to a 2D array
        train_curvs_array = np.array([train_curvs[key] for key in sorted(train_curvs.keys())])
        print("Shape of train_curvs_array:", train_curvs_array.shape)
        print('train_curvs_array:', train_curvs_array)

        results = np.array(train_curvs)
        metrics = dict(curvatures=results)
        np.savez(f'assets/proxy_results/loss_curvature_{args.dataset}_{args.arch}_s{args.seed}.npz', **metrics)


if __name__ == "__main__":
    main()
