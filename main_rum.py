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

    seed = args.seed
    unlearn = args.unlearn
    group_index = None
    num_indexes_to_replace = 1000


    # Define commands with dynamic mask parameters, below is an example for nothing-Finetune-SalUn RUM experiment
    runs = [
        # nothing-Finetune-SalUn (low-medium-high memorization order)
        f"python main_forget.py --seed {seed} --no_aug --sequential --mem mid --unlearn FT --unlearn_epochs 10 --unlearn_lr 0.1 --num_indexes_to_replace {num_indexes_to_replace} --class_to_replace {class_to_replace} --dataset {dataset} --arch {arch} --data {data} --epochs {epochs} --lr {lr} --decreasing_lr {decreasing_lr} --batch_size {batch_size} --mask '${original_model_path}'",
        f"python main_random.py --seed {seed} --no_aug --sequential --mem high --unlearn RL --unlearn_epochs 5 --unlearn_lr 0.1 --path '${saliency_map_path}' --num_indexes_to_replace {num_indexes_to_replace} --class_to_replace {class_to_replace} --dataset {dataset} --arch {arch} --data {data} --epochs {epochs} --lr {lr} --decreasing_lr {decreasing_lr} --batch_size {batch_size} --mask '${last_step_model_path}'",
        # evaluation
        f"python main_forget.py --seed {seed} --no_aug --unlearn seq_mix --mem mix --num_indexes_to_replace 3000 --class_to_replace {class_to_replace} --dataset {dataset} --arch {arch} --data {data} --epochs {epochs} --lr {lr} --decreasing_lr {decreasing_lr} --batch_size {batch_size} --mask '${final_model_path}'"

    ]

    for command in runs:
        run_command(command)


if __name__ == "__main__":
    main()
