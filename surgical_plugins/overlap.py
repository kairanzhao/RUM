import copy
import argparse
import os
from collections import OrderedDict
import shutil
import time
import numpy as np
import pandas as pd
import random
import re
import csv
import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet, ImageFolder
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, ConcatDataset, Subset
from matplotlib import pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.autonotebook import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn._reduction')

import utils
from utils import *
from scipy.stats import norm
from scipy.stats import pearsonr,spearmanr
import arg_parser
import unlearn.impl
from collections import OrderedDict
from trainer import train, validate

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

    # Update config only if the key doesn't exist
    # for key, value in vars(args).items():
    #     if key not in logger.config:
    #         logger.config[key] = value
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

def init_data_loader(data_loader, dataset, batch_size=128, seed=1, shuffle=True):
    # manual_seed(seed)
    setup_seed(seed)
    loader_args = {'num_workers': 0, 'pin_memory': False}
    def _init_fn(worker_id):
        np.random.seed(int(seed))
    return torch.utils.data.DataLoader(copy.deepcopy(dataset), batch_size=batch_size,num_workers=0,pin_memory=True,shuffle=shuffle)

def get_means(dataloader, model, args):
    # Register the hook to the penultimate layer
    t_FM = []
    hooks = []
    for hook_handle in hooks:
        hook_handle.remove()

    def hook1(model, input, output):
        t_FM.append(output)

    hooks.append(model.avgpool.register_forward_hook(hook1))

    model.to(args.device)

    overall_sums = None
    batch_count = 0
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(args.device)
            outputs = model(inputs)
            activations = t_FM[-1].reshape(len(outputs), -1)
            t_FM.clear()

            batch_mean = torch.mean(activations, dim=0).reshape(1, -1)
            if overall_sums is None:
                overall_sums = batch_mean
            else:
                overall_sums += batch_mean

            batch_count += 1

        for hook_handle in hooks:
            hook_handle.remove()

    overall_means = overall_sums / batch_count
    return overall_means
def get_norms_sum(dataloader, model, class_means, args):
    # Register the hook to the penultimate layer
    t_FM = []
    hooks = []
    for hook_handle in hooks:
        hook_handle.remove()
    def hook1(model, input, output):
        t_FM.append(output)

    #hooks.append(model.layer4[1].conv2.register_forward_hook(hook1))
    hooks.append(model.avgpool.register_forward_hook(hook1))
    model.to(args.device)
    overal_norms = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(args.device)
            outputs = model(inputs)
            activations = t_FM[-1].reshape(len(outputs),-1)
            # t_FM = []
            t_FM.clear()

            _norms = torch.sum(torch.norm(activations - class_means, p=2, dim=1) ** 2)
            overal_norms += _norms

        for hook_handle in hooks:
            hook_handle.remove()

    return overal_norms
def calculate_FC(model, retain_dataset, forget_dataset, args):
    r_dataset = copy.deepcopy(retain_dataset)

    retain_loader = init_data_loader(None, r_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=False)
    forget_loader = init_data_loader(None, forget_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=False) # batch_size=32

    retain_means = get_means(retain_loader, model, args)
    forget_means = get_means(forget_loader, model, args)

    # Calculate mu (mean of all outputs)
    mu = torch.mean(torch.cat((retain_means, forget_means), dim=0), dim=0)

    # Calculate RF for the batch
    C = 2  # Number of different classes: forget vs retain
    N1 = len(retain_dataset)
    N2 = len(forget_dataset)

    retain_norms_sum = get_norms_sum(retain_loader, model, retain_means, args)
    forget_norms_sum = get_norms_sum(forget_loader, model, forget_means, args)

    intra_class_norms = (retain_norms_sum / N1) + (forget_norms_sum / N2)

    norms_class_means = [torch.norm(class_means - mu, p=2, dim=1) ** 2 for class_means in [retain_means, forget_means]]
    inter_class_norms = torch.sum(torch.stack(norms_class_means)) / C

    FC = intra_class_norms / inter_class_norms

    print('FC_original: {}'.format(FC))
    print('RS_intra, FS_intra: {}, {}'.format((retain_norms_sum / N1), (forget_norms_sum / N2)))
    print('FC_intra_original: {}'.format(intra_class_norms))
    print('FC_inter_original: {}'.format(inter_class_norms))

    return FC, inter_class_norms

# Parameter space
def compute_diagonal_fisher_information(model, dataloader, criterion, device='cpu'):
    model = model.to(device)
    model.eval()

    diagonal_fisher = {name: torch.zeros_like(param, requires_grad=False) for name, param in model.named_parameters() if param.requires_grad}

    for inputs, targets in tqdm(dataloader):
        inputs = inputs.to(device)
        targets = torch.tensor([int(t) for t in targets]).to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        model.zero_grad()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                diagonal_fisher[name] += param.grad.data.pow(2)

    num_samples = len(dataloader.dataset)

    diagonal_fisher = {name: grad / num_samples for name, grad in diagonal_fisher.items()}

    return diagonal_fisher
def _get_items_with_prob_less_than_k(data, threshold):
    mu, std = norm.fit(data)
    pdf_values = norm.pdf(data, mu, std)
    selected_samples = np.where(pdf_values < threshold)[0]

    return selected_samples
def most_important_by_fisher(model, datasets, val_loader, frac, num_hist_bins, args, logger, files_to_save):
    retain_dataset, forget_dataset = datasets[0], datasets[1]

    f_loader = init_data_loader(None, forget_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=False)
    r_loader = init_data_loader(None, retain_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=False)

    FILM_f = compute_diagonal_fisher_information(model, f_loader, nn.CrossEntropyLoss(), args.device)
    FILM_r = compute_diagonal_fisher_information(model, r_loader, nn.CrossEntropyLoss(), args.device)
    FILM_f = FILM_f.detach().cpu().numpy()
    FILM_r = FILM_r.detach().cpu().numpy()
    """
    fim_f_min = torch.min(FILM_f)
    fim_f_max = torch.max(FILM_f)
    fim_r_min = torch.min(FILM_r)
    fim_r_max = torch.max(FILM_r)
    normalized_fisher_scores_f = (FILM_f - fim_f_min) / (fim_f_max - fim_f_min)
    normalized_fisher_scores_r = FILM_r - fim_r_min) / (fim_r_max - fim_r_min)
    sorted_indices_f = torch.argsort(normalized_fisher_scores_f, descending=True)
    sorted_indices_r = torch.argsort(normalized_fisher_scores_r, descending=True)
    num_params_to_select = int(len(FILM_f) * frac)
    top_params_indices_f = sorted_indices_f[:num_params_to_select]
    top_params_indices_r = sorted_indices_r[:num_params_to_select]
    """
    top_params_indices_f = _get_items_with_prob_less_than_k(FILM_f, frac)
    top_params_indices_r = _get_items_with_prob_less_than_k(FILM_r, frac)

    overlap = np.intersect1d(top_params_indices_f, top_params_indices_r)

    print('number of top 1% FIM retain', len(top_params_indices_r))
    print('number of top 1% FIM forget', len(top_params_indices_f))

    print('number of overlaps:', len(overlap))
    overlap_fraction = len(overlap) / min(len(top_params_indices_r), len(top_params_indices_f))
    print('overlap fraction:', overlap_fraction)

    # print('overlap indices locations:')
    # print(f"25th percentile: {int(np.percentile(overlap, 25))}, " +
    #       f"50th percentile: {int(np.percentile(overlap, 50))}, " +
    #       f"75th percentile: {int(np.percentile(overlap, 75))}, " +
    #       f"95th percentile: {int(np.percentile(overlap, 95))}")
    #
    # print('retain-set fisher scores (normalized):')
    # print(f"average: {np.mean(FILM_r)}" +
    #       f"median: {np.percentile(FILM_r, 50)}" +
    #       f"95th percentile: {np.percentile(FILM_r, 95)}")
    #
    # print('forget-set fisher scores (normalized):')
    # print(f"average: {np.mean(FILM_f)}" +
    #       f"median: {np.percentile(FILM_f, 50)}" +
    #       f"95th percentile: {np.percentile(FILM_f, 95)}")

    logger.log({'FIM overlap fraction': overlap_fraction})
    logger.log({f'overlap FIM': len(overlap)})
    logger.log({'number of top 1% FIM retain': len(top_params_indices_r)})
    logger.log({'number of top 1% FIM forget': len(top_params_indices_f)})
    logger.log({f'overlap indices 50th percentile FIM': int(np.percentile(overlap, 50))})
    logger.log({f'overlap indices 95th percentile FIM': int(np.percentile(overlap, 95))})
    logger.log({f'overlap indices 99th percentile FIM': int(np.percentile(overlap, 99))})
    logger.log({f'retain-set 50th percentile FIM': np.percentile(FILM_r, 50)})
    logger.log({f'retain-set 95th percentile FIM': np.percentile(FILM_r, 95)})
    logger.log({f'retain-set 99th percentile FIM': np.percentile(FILM_r, 99)})
    logger.log({f'forget-set 50th percentile FIM': np.percentile(FILM_f, 50)})
    logger.log({f'forget-set 95th percentile FIM': np.percentile(FILM_f, 95)})
    logger.log({f'forget-set 99th percentile FIM': np.percentile(FILM_f, 99)})

    fig1, axes1 = plt.subplots()
    fig2, axes2 = plt.subplots()

    sns.histplot(data=top_params_indices_r, bins=num_hist_bins, kde=False, alpha=1, label='retain-set top indices',
                 ax=axes1)
    sns.histplot(data=top_params_indices_f, bins=num_hist_bins, kde=False, alpha=0.9, label='forget-set top indices',
                 ax=axes1)
    sns.histplot(data=overlap, bins=num_hist_bins, kde=False, alpha=0.8, label=f'overlaps', ax=axes1)
    axes1.set_xlabel('parameters indices')

    sns.histplot(data=FILM_r, bins=num_hist_bins, kde=False, alpha=1, label='retain-set FILM', ax=axes2)
    sns.histplot(data=FILM_f, bins=num_hist_bins, kde=False, alpha=0.8, label='forget-set FILM', ax=axes2)
    axes2.set_xlabel('score values')

    axes1.legend(loc='upper right')

    axes1.set_title('Tope parameters based on FIM')
    fig1.tight_layout()
    fig1.savefig('assets/Plots/top_params_indice_distribution_FIM_fclass{}.png'.format(args.class_to_replace))
    files_to_save.append('assets/Plots/top_params_indice_distribution_FIM_fclass{}.png'.format(args.class_to_replace))

    axes2.legend(loc='upper right')
    axes2.set_title('Fisher scores distribution')
    fig2.tight_layout()
    fig2.savefig('assets/Plots/scores_distributions_FIM_fclass{}.png'.format(args.class_to_replace))
    files_to_save.append('assets/Plots/scores_distributions_FIM_fclass{}.png'.format(args.class_to_replace))

    return top_params_indices_r, top_params_indices_f, files_to_save, overlap_fraction

def main():
    global args, best_sa
    args = arg_parser.parse_args()
    print(args)

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")
    args.device = device
    os.makedirs(args.save_dir, exist_ok=True)

    args.wandb_group_name = f"{args.arch}-{args.dataset}-overlap"
    logger = wandb_init(args)
    files_to_save = []

    overlaps = []

    n_sampling = 300
    random.seed(42)
    unique_seeds = set()
    while len(unique_seeds) < n_sampling:
        unique_seeds.add(random.randint(1, 10000))
    unique_seeds = list(unique_seeds)

    overlap_compute = True
    if overlap_compute:
        for i, s in enumerate(unique_seeds):
            random.seed(s)
            seeds = [s]
            # n_class = random.randint(1, 10)
            # class_to_replace_configs = [sorted(random.sample(range(10), n_class))]
            class_to_replace_configs = [[0,1,2,3,4,5,6,7,8,9]]
            print(f'seed: {s}, class_to_replace: {class_to_replace_configs}')

            for class_to_replace in class_to_replace_configs:
                for seed in seeds:
                    args.seed = seed
                    args.class_to_replace = class_to_replace
                    if args.seed:
                        setup_seed(args.seed)
                    args.train_seed = args.seed

                    # prepare dataset
                    (
                        model,
                        train_loader_full,
                        val_loader,
                        test_loader,
                        marked_loader,
                    ) = utils.setup_model_dataset(args)
                    model = model.to(device)

                    def replace_loader_dataset(dataset, batch_size=args.batch_size, seed=1, shuffle=True):
                        utils.setup_seed(seed)
                        return torch.utils.data.DataLoader(
                            dataset,
                            batch_size=batch_size,
                            num_workers=0,
                            pin_memory=True,
                            shuffle=shuffle,
                        )
                    forget_dataset = copy.deepcopy(marked_loader.dataset)
                    if args.dataset == "svhn":
                        try:
                            marked = forget_dataset.targets < 0
                        except:
                            marked = forget_dataset.labels < 0
                        forget_dataset.data = forget_dataset.data[marked]
                        try:
                            forget_dataset.targets = -forget_dataset.targets[marked] - 1
                        except:
                            forget_dataset.labels = -forget_dataset.labels[marked] - 1
                        forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
                        print(len(forget_dataset))
                        retain_dataset = copy.deepcopy(marked_loader.dataset)
                        try:
                            marked = retain_dataset.targets >= 0
                        except:
                            marked = retain_dataset.labels >= 0
                        retain_dataset.data = retain_dataset.data[marked]
                        try:
                            retain_dataset.targets = retain_dataset.targets[marked]
                        except:
                            retain_dataset.labels = retain_dataset.labels[marked]
                        retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
                        print(len(retain_dataset))
                        assert len(forget_dataset) + len(retain_dataset) == len(
                            train_loader_full.dataset
                        )
                    else:
                        try:
                            marked = forget_dataset.targets < 0
                            forget_dataset.data = forget_dataset.data[marked]
                            forget_dataset.targets = -forget_dataset.targets[marked] - 1
                            forget_loader = replace_loader_dataset(
                                forget_dataset, seed=seed, shuffle=True
                            )
                            print(len(forget_dataset))
                            retain_dataset = copy.deepcopy(marked_loader.dataset)
                            marked = retain_dataset.targets >= 0
                            retain_dataset.data = retain_dataset.data[marked]
                            retain_dataset.targets = retain_dataset.targets[marked]
                            retain_loader = replace_loader_dataset(
                                retain_dataset, seed=seed, shuffle=True
                            )
                            print(len(retain_dataset))
                            assert len(forget_dataset) + len(retain_dataset) == len(
                                train_loader_full.dataset
                            )
                        except:
                            marked = forget_dataset.targets < 0
                            forget_dataset.imgs = forget_dataset.imgs[marked]
                            forget_dataset.targets = -forget_dataset.targets[marked] - 1
                            forget_loader = replace_loader_dataset(
                                forget_dataset, seed=seed, shuffle=True
                            )
                            print(len(forget_dataset))
                            retain_dataset = copy.deepcopy(marked_loader.dataset)
                            marked = retain_dataset.targets >= 0
                            retain_dataset.imgs = retain_dataset.imgs[marked]
                            retain_dataset.targets = retain_dataset.targets[marked]
                            retain_loader = replace_loader_dataset(
                                retain_dataset, seed=seed, shuffle=True
                            )
                            print(len(retain_dataset))
                            assert len(forget_dataset) + len(retain_dataset) == len(
                                train_loader_full.dataset
                            )

                    print(f"number of retain dataset {len(retain_dataset)}")
                    print(f"number of forget dataset {len(forget_dataset)}")
                    unique_classes, counts = np.unique(forget_dataset.targets, return_counts=True)
                    class_counts = dict(zip(unique_classes.tolist(), counts.tolist()))
                    print('forget set: ')
                    print(class_counts)

                    unlearn_data_loaders = OrderedDict(
                        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
                    )

                    print('Loading original model...')
                    filename = '{}_original_{}_bs{}_lr{}_seed{}_epochs{}.pth.tar'.format(
                        args.dataset,args.arch,args.batch_size, args.lr, args.seed, args.epochs)
                    pruning = state = 0
                    model_og = copy.deepcopy(model)
                    checkpoint = utils.load_checkpoint(device, args.save_dir, state, filename)
                    model_og.load_state_dict(checkpoint['state_dict'])
                    model_og.eval()


                    print('---------------Calculating Disentanglement Score---------------')
                    FC_original, FC_inter_original = calculate_FC(model_og, retain_dataset, forget_dataset, args)
                    FC_original = FC_original.item() if isinstance(FC_original, torch.Tensor) else FC_original
                    overlaps.append([seed, class_to_replace, class_counts, FC_original])

        with open('assets/sample_overlaps_{}_{}_num{}_eps{}_sample{}.csv'.format(
    args.dataset,args.arch,args.num_indexes_to_replace,args.epochs,n_sampling), 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['seed', 'class_to_replace', 'class_counts', 'overlap'])
            for result in overlaps:
                csvwriter.writerow(result)


if __name__ == "__main__":
    main()
