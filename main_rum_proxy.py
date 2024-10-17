import copy
import os
from collections import OrderedDict

import arg_parser
import evaluation
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, ConcatDataset, Subset
import unlearn
import utils
import numpy as np

# import pruner
from trainer import validate

from unlearn.impl import wandb_init, wandb_finish
from surgical_plugins.cluster import get_features, get_distance, get_fs, get_fs_dist_only
from surgical_plugins.overlap import calculate_FC, compute_diagonal_fisher_information

import subprocess

def run_command(command):
    subprocess.run(command, shell=True, check=True)

def main():
    args = arg_parser.parse_args()

    # Common parameters
    dataset = args.dataset
    arch = args.arch
    data = '/data/image_data'
    epochs = args.epochs
    lr = args.lr
    decreasing_lr = args.decreasing_lr
    batch_size = args.batch_size
    class_to_replace = -1
    proxy = args.mem_proxy

    seed = args.seed
    unlearn = args.unlearn
    group_index = None
    num_indexes_to_replace = 1000
    unlearn_step = args.unlearn_step

    # Define commands with dynamic mask parameters, below is an example for NegGrad+ --> NegGrad+ --> NegGrad+ RUM experiment (unlearning step 1)
    runs = [
        # RUM: NegGrad+ --> NegGrad+ --> NegGrad+ (low-medium-high memorization order or the corresponding proxy order)
        f"python main_forget.py --seed {seed} --no_aug --sequential --mem_proxy {proxy} --mem low --unlearn {unlearn} --unlearn_step {unlearn_step} --alpha 0.99 --unlearn_epochs 5 --unlearn_lr 0.01 --num_indexes_to_replace {num_indexes_to_replace} --class_to_replace {class_to_replace} --dataset {dataset} --arch {arch} --data {data} --epochs {epochs} --lr {lr} --decreasing_lr {decreasing_lr} --batch_size {batch_size} --mask '${last_step_model_path}'",
        f"python main_forget.py --seed {seed} --no_aug --sequential --mem_proxy {proxy} --mem mid --unlearn {unlearn} --unlearn_step {unlearn_step} --alpha 0.97 --unlearn_epochs 5 --unlearn_lr 0.01 --num_indexes_to_replace {num_indexes_to_replace} --class_to_replace {class_to_replace} --dataset {dataset} --arch {arch} --data {data} --epochs {epochs} --lr {lr} --decreasing_lr {decreasing_lr} --batch_size {batch_size} --mask '${last_step_model_path}'",
        f"python main_forget.py --seed {seed} --no_aug --sequential --mem_proxy {proxy} --mem high --unlearn {unlearn} --unlearn_step {unlearn_step} --alpha 0.97 --unlearn_epochs 10 --unlearn_lr 0.001 --num_indexes_to_replace {num_indexes_to_replace} --class_to_replace {class_to_replace} --dataset {dataset} --arch {arch} --data {data} --epochs {epochs} --lr {lr} --decreasing_lr {decreasing_lr} --batch_size {batch_size} --mask '${last_step_model_path}'",
        # Evaluation
        f"python main_forget.py --seed {seed} --no_aug --unlearn seq_mix --mem_proxy {proxy} --mem mix --num_indexes_to_replace 3000 --class_to_replace {class_to_replace} --dataset {dataset} --arch {arch} --data {data} --epochs {epochs} --lr {lr} --decreasing_lr {decreasing_lr} --batch_size {batch_size} --mask '${final_model_path}'"

     ]

    for command in runs:
        run_command(command)


if __name__ == "__main__":
    main()
