import copy
import argparse
import os
from collections import OrderedDict
from collections import defaultdict
import shutil
import time
import numpy as np
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
from scipy.stats import pearsonr, spearmanr, kendalltau
import pandas as pd
import arg_parser
import unlearn.impl
from collections import OrderedDict
from trainer import train, validate
from unlearn.impl import wandb_init, wandb_finish
import wandb

from sklearn.cluster import KMeans
from torchvision.models import resnet18
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import rbf_kernel

from surgical_plugins.overlap import calculate_FC

def get_features(dataloader, model, args):
    features_10d = []
    features = []
    labels = []
    original_indices = []

    t_FM = []
    hooks = []
    for hook_handle in hooks:
        hook_handle.remove()

    def hook1(model, input, output):
        t_FM.append(output)

    hooks.append(model.avgpool.register_forward_hook(hook1))

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            images, label = data
            images = images.to(args.device)
            label = label.to(args.device)
            outputs = model(images)

            activations = t_FM[-1].reshape(len(outputs), -1)
            t_FM.clear()

            features_10d.extend(outputs.cpu().numpy())
            features.extend(activations.cpu().numpy())
            labels.extend(label.cpu().numpy())
            global_indices = [idx * args.batch_size + i for i in range(len(images))]
            original_indices.extend(global_indices)

        for hook_handle in hooks:
            hook_handle.remove()
    return features, features_10d, labels, original_indices

def replace_loader_dataset(dataset, batch_size, seed=1, shuffle=True):
    utils.setup_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=shuffle,
    )

def get_DES(forget_dataset, retain_dataset, model, args,):

    f_loader = replace_loader_dataset(forget_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=True)
    r_loader = replace_loader_dataset(retain_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=True)
    f_features, _, _, f_original_indices = get_features(f_loader, model, args)
    r_features, _, _, r_original_indices = get_features(r_loader, model, args)
    f_features = np.array(f_features)
    r_features = np.array(r_features)
    # distance: FS --> RS centroid
    r_centroid = np.mean(r_features, axis=0)
    f_distances = np.linalg.norm(f_features - r_centroid, axis=1)
    f_distance_avg = np.mean(f_distances)
    # distance: RS --> FS centroid
    f_centroid = np.mean(f_features, axis=0)
    r_distances = np.linalg.norm(r_features - f_centroid, axis=1)
    r_distance_avg = np.mean(r_distances)

    distance_avg = (f_distance_avg + r_distance_avg) / 2

    print(f"Distance from FS to RS centroid: {f_distance_avg:.2f}")
    print(f"Distance from RS to FS centroid: {r_distance_avg:.2f}")
    print(f"Average Distance: {distance_avg:.2f}")

    return distance_avg


def get_mmd(forget_dataset, retain_dataset, model, args, kernel='rbf', gamma=1.0):
    f_loader = replace_loader_dataset(forget_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=True)
    r_loader = replace_loader_dataset(retain_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=True)
    f_features, _, _, f_original_indices = get_features(f_loader, model, args)
    r_features, _, _, r_original_indices = get_features(r_loader, model, args)
    f_features = np.array(f_features)
    r_features = np.array(r_features)

    if f_features.ndim != 2 or r_features.ndim != 2:
        raise ValueError("Feature arrays must be 2-dimensional")

    # Compute MMD
    if kernel == 'rbf':
        K_ff = rbf_kernel(f_features, f_features, gamma=gamma)
        K_rr = rbf_kernel(r_features, r_features, gamma=gamma)
        K_fr = rbf_kernel(f_features, r_features, gamma=gamma)

        m = f_features.shape[0]
        n = r_features.shape[0]
        mmd = np.sum(K_ff) / (m * m) + np.sum(K_rr) / (n * n) - 2 * np.sum(K_fr) / (m * n)

    else:
        raise ValueError(f"Invalid kernel: {kernel}")

    return mmd


def get_distance(train_loader, model_og, args, cluster_state=False, num_clusters=10, vis_cluster=False,
                 vis_distribution=False):
    # Extract features
    features, features_10d, gt_labels, original_indices = get_features(train_loader, model_og, args)
    features = np.array(features)

    if cluster_state:
        kmeans = KMeans(n_clusters=num_clusters, random_state=args.seed)
        cluster_labels = kmeans.fit_predict(features)
        labels = cluster_labels
    else:
        labels = np.array(gt_labels)

    overall_centroid = np.mean(features, axis=0)
    print(features.shape)

    centroids = []
    distances_matrix = []

    for i in range(num_clusters):
        cluster_features = features[labels == i]
        cluster_indices = np.array(original_indices)[labels == i]

        centroid = np.mean(cluster_features, axis=0)
        centroids.append(centroid)
        # Compute distances from centroid for each example in the cluster
        distances = np.linalg.norm(cluster_features - overall_centroid, axis=1)
        sorted_indices = np.argsort(distances)
        distances_matrix.append([(cluster_indices[j], distances[j]) for j in sorted_indices])

    return distances_matrix, features, gt_labels

def get_fs_dist_only(distances_matrix, train_loader, n_group, n_sample, group_index):
    forget_dataset_indices = set()
    distances = []
    fs_indices = []
    distances_group = []

    all_distances = [pair for sublist in distances_matrix for pair in sublist]
    all_distances.sort(key=lambda x: x[1])
    group_size = len(all_distances) // n_group   # cifar10 and cifar100
    # group_size = 1000                              # tiny-imagenet
    print(f"Group size: {group_size}")
    groups = [all_distances[i:i + group_size] for i in range(0, len(all_distances), group_size)]

    if groups and group_index < len(groups):
        # select first n_sample examples from each group
        selected_samples = groups[group_index][:n_sample]
        distances.extend([sample[1] for sample in selected_samples])
        fs_indices.extend([sample[0] for sample in selected_samples])
        forget_dataset_indices.update(fs_indices)

        median_distance = np.median(distances)
        mean_distance = np.mean(distances)
        print(f"Group {group_index}, Median Distance Across All Clusters: {median_distance}, Mean Distance: {mean_distance}")
        distances_group.append((median_distance, mean_distance))

    forget_dataset = torch.utils.data.Subset(train_loader.dataset, list(forget_dataset_indices))
    all_indices = set(range(len(train_loader.dataset)))
    retain_dataset_indices = all_indices - forget_dataset_indices
    retain_dataset = torch.utils.data.Subset(train_loader.dataset, list(retain_dataset_indices))
    # check the label distribution in the forget set
    forget_labels = [train_loader.dataset[i][1] for i in forget_dataset_indices]
    print('fs distribution', np.unique(forget_labels, return_counts=True))

    return forget_dataset, retain_dataset, fs_indices, distances_group

def get_fs_dist_only_mmd(distances_matrix, train_loader, n_group, n_sample, group_index):
    forget_dataset_indices = set()
    distances = []
    fs_indices = []
    distances_group = []

    all_distances = [pair for sublist in distances_matrix for pair in sublist]
    all_distances.sort(key=lambda x: x[1])
    group_size = len(all_distances) // n_group   # cifar10 and cifar100
    # group_size = 1000                              # tiny-imagenet
    print(f"Group size: {group_size}")
    groups = [all_distances[i:i + group_size] for i in range(0, len(all_distances), group_size)]

    if groups and group_index < len(groups):
        # select first n_sample examples from each group
        selected_samples = groups[group_index][:n_sample]
        distances.extend([sample[1] for sample in selected_samples])
        fs_indices.extend([sample[0] for sample in selected_samples])
        forget_dataset_indices.update(fs_indices)

        median_distance = np.median(distances)
        mean_distance = np.mean(distances)
        print(f"Group {group_index}, Median Distance Across All Clusters: {median_distance}, Mean Distance: {mean_distance}")
        distances_group.append((median_distance, mean_distance))

    forget_dataset = torch.utils.data.Subset(train_loader.dataset, list(forget_dataset_indices))
    all_indices = set(range(len(train_loader.dataset)))
    retain_dataset_indices = all_indices - forget_dataset_indices
    retain_dataset = torch.utils.data.Subset(train_loader.dataset, list(retain_dataset_indices))
    # check the label distribution in the forget set
    forget_labels = [train_loader.dataset[i][1] for i in forget_dataset_indices]
    print('fs distribution', np.unique(forget_labels, return_counts=True))

    return forget_dataset, retain_dataset, fs_indices, distances_group

def get_fs(distances_matrix, train_loader, n_group, n_sample, group_index):
    forget_dataset_indices = set()
    distances = []
    distances_group = []

    for cluster_idx, cluster in enumerate(distances_matrix):
        group_size = max(len(cluster) // n_group, 1)
        groups = [cluster[i:i + group_size] for i in range(0, len(cluster), group_size)]

        if groups and group_index < len(groups):
            selected_samples = groups[group_index][:n_sample]
            selected_sample_details = [(sample[0], sample[1]) for sample in selected_samples]  # (index, distance)
            distances.extend([sample[1] for sample in selected_samples])  # Collect distances

            for sample in selected_samples:
                original_idx, distance = sample
                forget_dataset_indices.add(original_idx)

    if distances:
        median_distance = np.median(distances)
        mean_distance = np.mean(distances)
        print(f"Group {group_index}, Median Distance Across All Clusters: {median_distance}, Mean Distance: {mean_distance}")
        print(f'max distances: {max(distances)}, min distances: {min(distances)}')
        distances_group.append((median_distance, mean_distance))

    forget_dataset = torch.utils.data.Subset(train_loader.dataset, list(forget_dataset_indices))
    all_indices = set(range(len(train_loader.dataset)))
    retain_dataset_indices = all_indices - forget_dataset_indices
    retain_dataset = torch.utils.data.Subset(train_loader.dataset, list(retain_dataset_indices))

    return forget_dataset, retain_dataset, forget_dataset_indices, distances_group

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
    files_to_save = []

    seeds = [1]
    class_to_replace_configs = [[-1]]
    group_ids = [3]

    if args.dataset == 'cifar10':
        num_clusters = 10
    elif args.dataset == 'cifar100':
        num_clusters = 100
    elif args.dataset == 'TinyImagenet':
        num_clusters = 200

    all_fs_indices = []
    all_distances = []
    distances_check = OrderedDict(((config,), []) for config in group_ids)

    for class_to_replace in class_to_replace_configs:
        for group_id in group_ids:
            args.group_index = group_id
            all_dfs = []
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
                    _,
                ) = utils.setup_model_dataset(args)
                model = model.to(device)

                train_loader = replace_loader_dataset(train_loader_full.dataset, batch_size=args.batch_size, seed=args.seed, shuffle=False)


                print('Loading original model...')
                filename = '0{}_original_{}_bs{}_lr{}_seed{}_epochs{}.pth.tar'.format(
                    args.dataset,args.arch,args.batch_size, args.lr, args.seed, args.epochs)
                pruning = state = 0
                model_og = copy.deepcopy(model)
                checkpoint = utils.load_checkpoint(device, args.save_dir, state, filename)
                model_og.load_state_dict(checkpoint['state_dict'])
                model_og.eval()

                distances_matrix, features, gt_labels = get_distance(train_loader, model_og, args, cluster_state=False, num_clusters=num_clusters,
                                                vis_cluster=False, vis_distribution=False)
                all_distances.append(distances_matrix)

                n_group = 15  # Number of groups in each cluster
                n_sample = 300  # Number of examples to select from each group
                n_sample_total = 3000
                fs, rs, fs_indices, dist_dict = get_fs_dist_only(distances_matrix, train_loader, n_group=n_group, n_sample=n_sample_total, group_index=group_id)
                # for MMD analysis
                # fs, rs, fs_indices, dist_dict = get_fs_dist_only_mmd(distances_matrix, train_loader, n_group=n_group, n_sample=n_sample_total, group_index=group_id)

                distances_check[(group_id,)].append(dist_dict)

                print('forget size', len(fs))
                # calculate_FC(model_og, rs, fs, args)       # get ES

                print('forget set indices', sorted(fs_indices))
                all_fs_indices.append(fs_indices)

                # MMD analysis
                mmd_square = get_mmd(fs, rs, model_og, args, kernel='rbf', gamma=1.0)
                mmd = np.sqrt(mmd_square)

                print(f'[Group {group_id}, seed {seed}] MMD:', mmd)


if __name__ == "__main__":
    main()
