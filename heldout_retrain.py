import os
import numpy as np
import yaml
from tqdm import tqdm
import argparse
import copy
import shutil
from rich import print as rich_print
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import time

import arg_parser
import utils
from utils import *
import wandb

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

def calculate_kl(p, q):
    kl_divergence = (p * (torch.log(p) - torch.log(q))).sum(dim=1) + (q * (torch.log(q) - torch.log(p))).sum(dim=1)
    kl_divergence = kl_divergence / 2
    return kl_divergence


def finetune(model, train_loader, args, device, logger=None):
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.ft_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # lr_dict = {'learning_rate': args.ft_lr, 'lr_decay_epochs': args.lr_decay_epochs, 'lr_decay_rate': args.lr_decay_rate}

    best_loss = float('inf')
    epochs_without_improvement = 0
    for epoch in range(args.ft_epochs):
        # adjust_learning_rate(epoch, lr_dict, optimizer)
        running_loss = 0
        correct_predictions = 0
        total_predictions = 0
        for i, (images, labels, unique_ids) in enumerate(train_loader):
            images, labels, unique_ids = images.to(device), labels.to(device), unique_ids.to(device)
            model.train()
            optimizer.zero_grad()
            outputs = model(images)
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

        if running_loss < best_loss:
            best_loss = running_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= 4:
            print(f"No improvement in loss for {epochs_without_improvement} epochs. Stopping training.")
            break

        training_accuracy = 100 * correct_predictions / total_predictions
        print(
            f"Epoch {epoch + 1}/{args.ft_epochs}, lr:{args.ft_lr}, Loss: {running_loss / len(train_loader):.4f}, Training Accuracy: {training_accuracy:.2f}%")

    print('Finetuning finished!')

    return 0


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


def retrain_divergences(original_model, retrain_model, train_loader, args, device):
    train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=args.batch_size, shuffle=False)

    original_model.eval()
    retrain_model.eval()
    kl_divergences = []
    unique_ids_list = []

    epsilon = 1e-10
    with torch.no_grad():
        for images, labels, unique_ids in train_loader:
            images = images.to(device)
            orig_probs = torch.softmax(original_model(images), dim=1)
            retrain_probs = torch.softmax(retrain_model(images), dim=1)

            orig_probs = torch.clamp(orig_probs, min=epsilon)
            retrain_probs = torch.clamp(retrain_probs, min=epsilon)

            kl_divergence = calculate_kl(orig_probs, retrain_probs)
            kl_divergences.extend(kl_divergence.cpu().numpy().flatten())
            unique_ids_list.extend(unique_ids.cpu().numpy().flatten())

            if np.isnan(kl_divergence.cpu().numpy()).any():
                print("nan in kl_divergence")
                nan_indices = np.where(np.isnan(kl_divergence.cpu().numpy()))[0]
                unique_ids = unique_ids.cpu().numpy().flatten()
                unique_id = unique_ids[nan_indices]
                print("unique_id: ", unique_id)
                orig_probs = orig_probs.cpu().numpy()
                retrain_probs = retrain_probs.cpu().numpy()
                orig_prob = orig_probs[nan_indices]
                retrain_prob = retrain_probs[nan_indices]
                print("orig_prob: ", orig_prob)
                print("retrain_prob: ", retrain_prob)

    kl_divergences = np.array(kl_divergences)
    unique_ids_list = np.array(unique_ids_list)
    kl_with_ids = np.column_stack((unique_ids_list, kl_divergences))

    return kl_with_ids


def main():
    args = arg_parser.parse_args()
    rich_print(args)

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)

    if args.seed:
        setup_seed(args.seed)

    logger = wandb_init(args)
    files_to_save = []

    if args.dataset == "TinyImagenet":
        args.data_dir = "/data/image_data/tiny-imagenet-200/"
        print(args.data_dir)

    # prepare dataset
    original_dataset = args.dataset
    args.dataset = f"{original_dataset}withids"
    (
        model,
        train_loader,
        val_loader,
        test_loader,
        marked_loader,
    ) = setup_model_dataset(args)
    args.dataset = original_dataset
    model.cuda()

    print(f"number of train dataset {len(train_loader.dataset)}")
    print(f"number of val dataset {len(val_loader.dataset)}")

    if args.unlearn is not None:
        save_dir_rt = f'assets/unlearn/{args.unlearn}'
        if args.mem_proxy is not None:
            if args.unlearn_step == 1:
                if args.unlearn == 'seq_mix':
                    filename_rt = (f'{args.unlearn}_{args.uname}_{args.dataset}_{args.arch}_{args.class_to_replace}_num{args.num_indexes_to_replace}_'
                                   f'groupid{args.group_index}_proxy{args.mem_proxy}_{args.mem}_seed{args.seed}.pth.tar')
                else:
                    filename_rt = (f'{args.unlearn}_{args.dataset}_{args.arch}_{args.class_to_replace}_num{args.num_indexes_to_replace}_'
                               f'groupid{args.group_index}_proxy{args.mem_proxy}_{args.mem}_seed{args.seed}.pth.tar')
            else:
                if args.unlearn == 'seq_mix':
                    filename_rt = (f'{args.unlearn}_{args.uname}_{args.dataset}_{args.arch}_{args.class_to_replace}_num{args.num_indexes_to_replace}_'
                                   f'groupid{args.group_index}_proxy{args.mem_proxy}_{args.mem}_step{args.unlearn_step}_seed{args.seed}.pth.tar')
                else:
                    filename_rt = (f'{args.unlearn}_{args.dataset}_{args.arch}_{args.class_to_replace}_num{args.num_indexes_to_replace}_'
                               f'groupid{args.group_index}_proxy{args.mem_proxy}_{args.mem}_step{args.unlearn_step}_seed{args.seed}.pth.tar')
        else:
            filename_rt = '{}_{}_{}_{}_num{}_groupid{}_mem{}_seed{}.pth.tar'.format(args.unlearn,
                args.dataset, args.arch, args.class_to_replace, args.num_indexes_to_replace, args.group_index,
                args.mem, args.seed)
        print('check 3, which model to load: ', filename_rt)
        checkpoint = utils.load_checkpoint(device, save_dir_rt, 'retrain', filename=filename_rt)

    else:
        args.mask = 'assets/checkpoints/0{}_original_{}_bs256_lr0.1_seed{}_epochs{}.pth.tar'.format(
                    args.dataset, args.arch, args.seed, args.epochs)
        print('check 3, which model to load: ', args.mask)
        checkpoint = torch.load(args.mask, map_location=device)
    if "state_dict" in checkpoint.keys():
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=False)
    print("Checkpoint loaded successfully")


    retrain = copy.deepcopy(model)
    print('Start finetuning...')
    start_1 = time.time()
    finetune(retrain, test_loader, args, device, logger)
    end_1 = time.time()

    orig_train_accuracy = evaluate_model(model, train_loader, device)
    orig_test_accuracy = evaluate_model(model, test_loader, device)

    ret_train_accuracy = evaluate_model(retrain, train_loader, device)
    ret_test_accuracy = evaluate_model(retrain, test_loader, device)

    print('[Original Model]')
    print(f"Train accuracy: {orig_train_accuracy:.2f}%, Test accuracy: {orig_test_accuracy:.2f}%")
    print('[Retrain Model]')
    print(f"Train accuracy: {ret_train_accuracy:.2f}%, Test accuracy: {ret_test_accuracy:.2f}%")

    start_2 = time.time()
    kl_with_ids = retrain_divergences(model, retrain, train_loader, args, device)
    end_2 = time.time()
    overhead = end_1 - start_1 + end_2 - start_2
    print(f"Time taken for Heldout Retrain: {overhead:.3f} seconds")

    print("KL Divergences shape: ", kl_with_ids.shape)

    metrics = dict(kl_divergences=kl_with_ids)

    if args.unlearn is not None:
        if args.unlearn == 'seq_mix':
            np.savez(f'assets/proxy_results/heldout_retrain_{args.dataset}_{args.arch}_{args.unlearn}_{args.uname}_step{args.unlearn_step}_num{args.num_indexes_to_replace}_groupid{args.group_index}_proxy{args.mem_proxy}_{args.mem}_s{args.seed}.npz', **metrics)
        else:
            np.savez(f'assets/proxy_results/heldout_retrain_{args.dataset}_{args.arch}_{args.unlearn}_step{args.unlearn_step}_num{args.num_indexes_to_replace}_groupid{args.group_index}_proxy{args.mem_proxy}_{args.mem}_s{args.seed}.npz', **metrics)
    else:
        np.savez(f'assets/proxy_results/heldout_retrain_{args.dataset}_{args.arch}_s{args.seed}.npz', **metrics)


if __name__ == '__main__':
    main()