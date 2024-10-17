import copy
import argparse
import os
from collections import OrderedDict
from collections import defaultdict
import shutil
import time
import numpy as np
import pandas as pd
import scipy.stats as stats
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

import utils
from utils import *
from scipy.stats import norm
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import arg_parser
import unlearn.impl
from collections import OrderedDict
from trainer import train, validate
from unlearn.impl import wandb_init, wandb_finish
import wandb

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn._reduction')
pd.set_option('display.max_columns', None)


def extract_weights(model):
    model_weights = []
    for param in model.parameters():
        model_weights.append(param.detach().cpu().numpy())
    # Flatten and concatenate all weights into a single vector
    return np.concatenate([w.flatten() for w in model_weights])


def euclidean_distance(model1, model2):
    weights1 = extract_weights(model1)
    weights2 = extract_weights(model2)
    l2_distance = np.sqrt(np.sum((weights1 - weights2) ** 2))
    return l2_distance

def cosine_similarity(model1, model2):
    weights1 = extract_weights(model1)
    weights2 = extract_weights(model2)
    cosine_similarity = np.dot(weights1, weights2) / (np.linalg.norm(weights1) * np.linalg.norm(weights2))
    return cosine_similarity

def calculate_confidence_interval(data):
    mean = np.mean(data)
    ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=stats.sem(data))
    # preserve 3 decimal points
    return mean, ci

def main():
    # global args, best_sa
    args = arg_parser.parse_args()
    print(args)

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")
    args.device = device
    os.makedirs(args.save_dir, exist_ok=True)

    args.wandb_group_name = f"{args.arch}-{args.dataset}-cluster"
    logger = wandb_init(args)

    seeds = [1]
    args.class_to_replace = [-1]
    group_ids = [0]

    df_results = pd.DataFrame(columns=['Group ID', 'L2 Distance avg', 'L2 Distance CI',
                                       'Cosine Similarity avg', 'Cosine Similarity CI'])
    for group_id in group_ids:
        args.group_index = group_id
        results = {'Group ID':[], 'Seed':[], 'L2 Distance': [], 'Cosine Similarity': []}
        for seed in seeds:
            args.seed = seed
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
            # non shuffled train loader to trace the index
            train_loader = DataLoader(
                train_loader_full.dataset,
                batch_size=args.batch_size,
                shuffle=False
            )

            print('Loading original model...')
            filename = '0{}_original_{}_bs{}_lr{}_seed{}_epochs{}.pth.tar'.format(
                args.dataset, args.arch, args.batch_size, args.lr, seed, args.epochs)
            pruning = state = 0
            model_og = copy.deepcopy(model)
            checkpoint = utils.load_checkpoint(device, 'assets/checkpoints', state, filename)
            model_og.load_state_dict(checkpoint['state_dict'])
            model_og.eval()

            filename = '0{}_original_{}_bs{}_lr{}_seed{}_epochs{}.pth.tar'.format(
                args.dataset, args.arch, args.batch_size, args.lr, 2, args.epochs)
            pruning = state = 0
            model_og2 = copy.deepcopy(model)
            checkpoint = utils.load_checkpoint(device, 'assets/checkpoints', state, filename)
            model_og2.load_state_dict(checkpoint['state_dict'])
            model_og2.eval()

            print(f'Loading retrained model...')
            filename = 'retrain_{}_{}_{}_num{}_groupid{}_seed{}.pth.tar'.format(args.dataset, args.arch,
                        args.class_to_replace, args.num_indexes_to_replace, args.group_index, seed)
            model_rt = copy.deepcopy(model)
            checkpoint = utils.load_checkpoint(device, 'assets/unlearn/retrain', 'retrain', filename=filename)
            model_rt.load_state_dict(checkpoint['state_dict'])
            model_rt.eval()

            print(f'Loading unlearned model:{args.unlearn}...')
            if args.unlearn is not None:
                if args.unlearn == 'RL':
                    args.save_dir = 'assets/unlearn/salun'
                else:
                    args.save_dir = f'assets/unlearn/{args.unlearn}'
            filename = '{}_{}_{}_{}_num{}_groupid{}_seed{}.pth.tar'.format(args.unlearn, args.dataset, args.arch,
                        args.class_to_replace, args.num_indexes_to_replace, args.group_index, seed)
            model_u = copy.deepcopy(model)
            checkpoint = utils.load_checkpoint(device, args.save_dir, args.unlearn, filename=filename)
            model_u.load_state_dict(checkpoint['state_dict'])
            model_u.eval()

            l2_dist = euclidean_distance(model_u, model_og)
            cos_sim = cosine_similarity(model_u, model_og)
            print('l2 distance:', l2_dist)
            print('cosine similarity:', cos_sim)
            results['Group ID'].append(group_id)
            results['Seed'].append(f'{seed}-{seed}')
            results['L2 Distance'].append(l2_dist)
            results['Cosine Similarity'].append(cos_sim)

        avg_l2, ci_l2 = calculate_confidence_interval(results['L2 Distance'])
        avg_cos, ci_cos = calculate_confidence_interval(results['Cosine Similarity'])

        df_results = df_results.append({
            'Group ID': group_id,
            'L2 Distance avg': avg_l2,
            'L2 Distance CI': ci_l2,
            'Cosine Similarity avg': avg_cos,
            'Cosine Similarity CI': ci_cos
        }, ignore_index=True)

    # print(df_results)
    print(df_results.to_csv(sep='\t', index=False))


if __name__ == "__main__":
    main()
