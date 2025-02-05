import copy
import argparse
import os
import pdb
import pickle
import random
import shutil
import time
from copy import deepcopy
from rich import print as rich_print
from collections import defaultdict

import arg_parser
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, ConcatDataset, Subset
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from pruner import *
from torch.utils.data.sampler import SubsetRandomSampler
from trainer import train, validate
from trainer.val import validate_withids
import utils
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

def calculate_negative_entropy(probs):
    return (probs * torch.log(probs)).sum().item()


def main():
    global args, best_sa
    args = arg_parser.parse_args()
    rich_print(args)

    torch.cuda.set_device(int(args.gpu))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        setup_seed(args.seed)
        seed = args.seed

    logger = wandb_init(args)
    files_to_save = []

    if args.dataset == "TinyImagenetwithids" or args.dataset == "TinyImagenet":
        args.data_dir = "/data/image_data/tiny-imagenet-200/"
        print(args.data_dir)

    # prepare dataset
    if args.unlearn is not None:
        original_dataset = args.dataset
        args.dataset = f"{original_dataset}withids"
    else:
        pass
    (
        model,
        train_loader,
        val_loader,
        test_loader,
        marked_loader,
    ) = setup_model_dataset(args)
    if args.unlearn is not None:
        args.dataset = original_dataset
    else:
        pass
    model.cuda()

    print(f"number of train dataset {len(train_loader.dataset)}")
    print(f"number of val dataset {len(val_loader.dataset)}")

    ###### begin comment out for sequential unlearning ######
    # def replace_loader_dataset(dataset, batch_size=args.batch_size, seed=1, shuffle=True):
    #     utils.setup_seed(seed)
    #     return torch.utils.data.DataLoader(
    #         dataset,
    #         batch_size=batch_size,
    #         num_workers=0,
    #         pin_memory=True,
    #         shuffle=shuffle,
    #     )
    #
    # forget_dataset = copy.deepcopy(marked_loader.dataset)
    #
    # if args.mem is not None and args.group_index is None and args.mem_proxy is None:
    #     fine_overlap = False
    #     mem_fs_split = True
    #     proxy_fs_split = False
    # elif args.mem is None and args.group_index is not None and args.mem_proxy is None:
    #     fine_overlap = True
    #     mem_fs_split = False
    #     proxy_fs_split = False
    # elif args.mem_proxy is not None:
    #     fine_overlap = False
    #     mem_fs_split = False
    #     proxy_fs_split = True
    # else:
    #     fine_overlap = False
    #     mem_fs_split = False
    #     proxy_fs_split = False
    #
    # if fine_overlap:
    #     print('[fs split]: fine_overlap')
    #     train_loader = DataLoader(
    #         train_loader_full.dataset,
    #         batch_size=args.batch_size,
    #         shuffle=False
    #     )
    #
    #     print('Loading original model...')
    #     filename = '0{}_original_{}_bs{}_lr{}_seed{}_epochs{}.pth.tar'.format(
    #         args.dataset, args.arch, args.batch_size, args.lr, args.seed, args.epochs)
    #     pruning = state = 0
    #     model_og = copy.deepcopy(model)
    #     checkpoint = utils.load_checkpoint(device, 'assets/checkpoints', state, filename)
    #     model_og.load_state_dict(checkpoint['state_dict'])
    #     model_og.eval()
    #
    #     if args.dataset == 'cifar10':
    #         num_clusters = 10
    #         n_group = 15
    #     elif args.dataset == 'cifar100':
    #         num_clusters = 100
    #         n_group = 15
    #     elif args.dataset == 'TinyImagenet':
    #         num_clusters = 200
    #         n_group = 100
    #     distances_matrix,_,_ = get_distance(train_loader, model_og, args, cluster_state=False, num_clusters=num_clusters,
    #                                     vis_cluster=False, vis_distribution=False)
    #
    #
    #     n_sample = args.num_indexes_to_replace
    #     if args.dataset == 'cifar10' or args.dataset == 'cifar100':
    #         _, _, l_des_idx, _ = get_fs_dist_only(distances_matrix, train_loader, n_group=n_group, n_sample=n_sample, group_index=3)
    #         _, _, m_des_idx, _ = get_fs_dist_only(distances_matrix, train_loader, n_group=n_group, n_sample=n_sample, group_index=12)
    #         _, _, h_des_idx, _ = get_fs_dist_only(distances_matrix, train_loader, n_group=n_group, n_sample=n_sample, group_index=14)
    #     elif args.dataset == 'TinyImagenet':
    #         _, _, l_des_idx_1, _ = get_fs_dist_only(distances_matrix, train_loader, n_group=n_group, n_sample=n_sample, group_index=0)
    #         _, _, l_des_idx_2, _ = get_fs_dist_only(distances_matrix, train_loader, n_group=n_group, n_sample=n_sample, group_index=1)
    #         _, _, l_des_idx_3, _ = get_fs_dist_only(distances_matrix, train_loader, n_group=n_group, n_sample=n_sample, group_index=2)
    #         _, _, m_des_idx_1, _ = get_fs_dist_only(distances_matrix, train_loader, n_group=n_group, n_sample=n_sample, group_index=49)
    #         _, _, m_des_idx_2, _ = get_fs_dist_only(distances_matrix, train_loader, n_group=n_group, n_sample=n_sample, group_index=50)
    #         _, _, m_des_idx_3, _ = get_fs_dist_only(distances_matrix, train_loader, n_group=n_group, n_sample=n_sample, group_index=51)
    #         _, _, h_des_idx_1, _ = get_fs_dist_only(distances_matrix, train_loader, n_group=n_group, n_sample=n_sample, group_index=97)
    #         _, _, h_des_idx_2, _ = get_fs_dist_only(distances_matrix, train_loader, n_group=n_group, n_sample=n_sample, group_index=98)
    #         _, _, h_des_idx_3, _ = get_fs_dist_only(distances_matrix, train_loader, n_group=n_group, n_sample=n_sample, group_index=99)
    #         l_des_idx = l_des_idx_1 + l_des_idx_2 + l_des_idx_3
    #         m_des_idx = m_des_idx_1 + m_des_idx_2 + m_des_idx_3
    #         h_des_idx = h_des_idx_1 + h_des_idx_2 + h_des_idx_3
    #
    #
    #     print('check: Group index: ', args.group_index)
    #     if args.group_index == 3:
    #         forget_dataset_indices = l_des_idx
    #     elif args.group_index == 12:
    #         forget_dataset_indices = m_des_idx
    #     elif args.group_index == 14:
    #         forget_dataset_indices = h_des_idx
    #     elif args.group_index == -1:
    #         ld = l_des_idx[:args.num_indexes_to_replace // 3]
    #         md = m_des_idx[:args.num_indexes_to_replace // 3]
    #         hd = h_des_idx[:args.num_indexes_to_replace // 3]
    #         forget_dataset_indices = ld + md + hd
    #     else:
    #         raise ValueError('Invalid des value')
    #
    #     forget_dataset = torch.utils.data.Subset(train_loader.dataset, list(forget_dataset_indices))
    #     all_indices = set(range(len(train_loader.dataset)))
    #     if args.sequential:
    #         retain_dataset_indices = all_indices - set(h_des_idx + m_des_idx + l_des_idx)
    #     else:
    #         retain_dataset_indices = all_indices - set(forget_dataset_indices)
    #
    #     retain_dataset = torch.utils.data.Subset(train_loader.dataset, list(retain_dataset_indices))
    #     forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
    #     retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
    #
    #     hd = list(h_des_idx)[:1000]
    #     md = list(m_des_idx)[:1000]
    #     ld = list(l_des_idx)[:1000]
    #     hd_dataset = torch.utils.data.Subset(train_loader.dataset, list(hd))
    #     md_dataset = torch.utils.data.Subset(train_loader.dataset, list(md))
    #     ld_dataset = torch.utils.data.Subset(train_loader.dataset, list(ld))
    #     hd_loader = replace_loader_dataset(hd_dataset, seed=seed, shuffle=True)
    #     md_loader = replace_loader_dataset(md_dataset, seed=seed, shuffle=True)
    #     ld_loader = replace_loader_dataset(ld_dataset, seed=seed, shuffle=True)
    # elif mem_fs_split:
    #     print('[fs split]: mem_fs_split')
    #     train_loader = DataLoader(
    #         train_loader_full.dataset,
    #         batch_size=args.batch_size,
    #         shuffle=False
    #     )
    #
    #     if args.dataset == 'cifar10':
    #         loaded_results = np.load('estimates_results_woShuffle.npz')
    #         loaded_memorization = loaded_results['memorization']
    #     elif args.dataset == 'cifar100':
    #         loaded_results = np.load('cifar100_infl_matrix.npz')
    #         loaded_memorization = loaded_results['tr_mem']
    #     loaded_memorization = loaded_memorization[train_idx]
    #
    #     indices = list(range(len(train_loader.dataset)))
    #     indices_mem = list(zip(indices, loaded_memorization))
    #
    #     indices_mem.sort(key=lambda x: x[1], reverse=True)
    #     h_mem_list = indices_mem[:args.num_indexes_to_replace]
    #     l_mem_list = indices_mem[-args.num_indexes_to_replace:]
    #     indices_mem.sort(key=lambda x: abs(x[1] - 0.5))
    #     m_mem_list = indices_mem[:args.num_indexes_to_replace]
    #
    #     if args.shuffle:
    #         indices_mem_mix = h_mem_list + l_mem_list + m_mem_list
    #         np.random.shuffle(indices_mem_mix)
    #         h_mem_list = indices_mem_mix[:args.num_indexes_to_replace]
    #         l_mem_list = indices_mem_mix[-args.num_indexes_to_replace:]
    #         m_mem_list = indices_mem_mix[args.num_indexes_to_replace:-args.num_indexes_to_replace]
    #     else:
    #         pass
    #
    #     h_mem_idx, h_mem = zip(*h_mem_list)
    #     l_mem_idx, l_mem = zip(*l_mem_list)
    #     m_mem_idx, m_mem = zip(*m_mem_list)
    #     print('check: h_mem: ', h_mem[:100])
    #     print('check: l_mem: ', l_mem[:100])
    #     print('check: m_mem: ', m_mem[:100])
    #
    #     print('check: args.mem: ', args.mem)
    #     if args.mem == 'high':
    #         forget_dataset_indices = h_mem_idx
    #     elif args.mem == 'low':
    #         forget_dataset_indices = l_mem_idx
    #     elif args.mem == 'mid':
    #         forget_dataset_indices = m_mem_idx
    #     elif args.mem == 'mix':
    #         hm = h_mem_idx[:args.num_indexes_to_replace // 3]
    #         mm = m_mem_idx[:args.num_indexes_to_replace // 3]
    #         lm = l_mem_idx[-args.num_indexes_to_replace // 3:]
    #         forget_dataset_indices = hm + mm + lm
    #     else:
    #         raise ValueError('Invalid mem value')
    #
    #     forget_dataset = torch.utils.data.Subset(train_loader.dataset, list(forget_dataset_indices))
    #     all_indices = set(range(len(train_loader.dataset)))
    #     if args.sequential:
    #         if args.mem == 'low':
    #             retain_dataset_indices = all_indices - set(l_mem_idx)
    #         elif args.mem == 'mid':
    #             retain_dataset_indices = all_indices - set(l_mem_idx + m_mem_idx)
    #         elif args.mem == 'high':
    #             retain_dataset_indices = all_indices - set(l_mem_idx + m_mem_idx + h_mem_idx)
    #
    #         print('check 2, retain set size: ', len(retain_dataset_indices))
    #     else:
    #         retain_dataset_indices = all_indices - set(forget_dataset_indices)
    #         print('check 2, retain set size: ', len(retain_dataset_indices))
    #
    #     retain_dataset = torch.utils.data.Subset(train_loader.dataset, list(retain_dataset_indices))
    #     forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
    #     retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
    #
    #     hm = h_mem_idx[:1000]
    #     lm = l_mem_idx[-1000:]
    #     mm = m_mem_idx[:1000]
    #     hm_dataset = torch.utils.data.Subset(train_loader.dataset, list(hm))
    #     mm_dataset = torch.utils.data.Subset(train_loader.dataset, list(mm))
    #     lm_dataset = torch.utils.data.Subset(train_loader.dataset, list(lm))
    #     hm_loader = replace_loader_dataset(hm_dataset, seed=seed, shuffle=True)
    #     mm_loader = replace_loader_dataset(mm_dataset, seed=seed, shuffle=True)
    #     lm_loader = replace_loader_dataset(lm_dataset, seed=seed, shuffle=True)
    # elif proxy_fs_split:
    #     print(f'[fs split]: proxy of memorization - {args.mem_proxy}')
    #     if args.mem_proxy == 'confidence' or args.mem_proxy == 'max_conf' or args.mem_proxy == 'entropy' or args.mem_proxy == 'bi_acc':
    #         loaded_results = np.load(f'assets/proxy_results/event_results_{args.dataset}withids_{args.arch}_s{seed}.npz',
    #                                  allow_pickle=True)
    #         loaded_events = loaded_results['events']
    #         sorted_events = loaded_events[loaded_events[:, 0].argsort()]
    #
    #         nan_or_inf_indices = np.isnan(sorted_events) | np.isinf(sorted_events)
    #         print("Number of NaNs or infinities:", np.sum(nan_or_inf_indices))
    #         sorted_events = np.nan_to_num(sorted_events, nan=np.inf)
    #
    #         if args.mem_proxy == 'confidence':
    #             sorted_confidences = sorted_events[:, 2]
    #         elif args.mem_proxy == 'max_conf':
    #             sorted_confidences = sorted_events[:, 3]
    #         elif args.mem_proxy == 'entropy':
    #             sorted_confidences = sorted_events[:, 4]
    #         elif args.mem_proxy == 'bi_acc':
    #             sorted_confidences = sorted_events[:, 5]
    #
    #         indices = sorted_events[:, 0].astype(int)
    #         indices_con = list(zip(indices, sorted_confidences))
    #         indices_con.sort(key=lambda x: x[1], reverse=True)
    #
    #         if args.mem_proxy == 'entropy':
    #             indices_con = [x for x in indices_con if not np.isinf(x[1])]
    #             print('check: number of values left after removing nan or inf: ', len(indices_con))
    #
    #         h_con_list = indices_con[:args.num_indexes_to_replace]
    #         l_con_list = indices_con[-args.num_indexes_to_replace:]
    #         median_con = np.median(sorted_confidences)
    #         medium_con = (np.min(sorted_confidences) + np.max(sorted_confidences)) / 2
    #         print('check: min confidence: ', np.min(sorted_confidences), 'max confidence: ', np.max(sorted_confidences), 'medium confidence: ', medium_con, 'median confidence: ', median_con)
    #         indices_con.sort(key=lambda x: abs(x[1] - medium_con))
    #         m_con_list = indices_con[:args.num_indexes_to_replace]
    #
    #         if args.shuffle:
    #             indices_proxy_mix = h_con_list + l_con_list + m_con_list
    #             np.random.shuffle(indices_proxy_mix)
    #             h_con_list = indices_proxy_mix[:args.num_indexes_to_replace]
    #             l_con_list = indices_proxy_mix[-args.num_indexes_to_replace:]
    #             m_con_list = indices_proxy_mix[args.num_indexes_to_replace:-args.num_indexes_to_replace]
    #         else:
    #             pass
    #
    #         h_con_idx, h_con = zip(*h_con_list)
    #         l_con_idx, l_con = zip(*l_con_list)
    #         m_con_idx, m_con = zip(*m_con_list)
    #
    #         print(f'check: h_con [{min(h_con):.3f}, {max(h_con):.3f}], examples: {h_con[:10]}')
    #         print(f'check: m_con [{min(m_con):.3f}, {max(m_con):.3f}], examples: {m_con[:10]}')
    #         print(f'check: l_con [{min(l_con):.3f}, {max(l_con):.3f}], examples: {l_con[:10]}')
    #
    #         if args.mem == 'high':
    #             forget_dataset_indices = h_con_idx
    #         elif args.mem == 'low':
    #             forget_dataset_indices = l_con_idx
    #         elif args.mem == 'mid':
    #             forget_dataset_indices = m_con_idx
    #         elif args.mem == 'mix':
    #             hc = h_con_idx[:args.num_indexes_to_replace // 3]
    #             mc = m_con_idx[:args.num_indexes_to_replace // 3]
    #             lc = l_con_idx[-args.num_indexes_to_replace // 3:]
    #             forget_dataset_indices = hc + mc + lc
    #         else:
    #             raise ValueError('Invalid mem value')
    #
    #         forget_images = []
    #         forget_labels = []
    #         forget_unique_ids = []
    #         retain_images = []
    #         retain_labels = []
    #         retain_unique_ids = []
    #
    #         for images, labels, unique_ids in train_loader:
    #             for idx, unique_id in enumerate(unique_ids):
    #                 if unique_id.item() in forget_dataset_indices:
    #                     forget_images.append(images[idx])
    #                     forget_labels.append(labels[idx])
    #                     forget_unique_ids.append(unique_id)
    #                 else:
    #                     retain_images.append(images[idx])
    #                     retain_labels.append(labels[idx])
    #                     retain_unique_ids.append(unique_id)
    #
    #         # Convert lists to tensors
    #         forget_images = torch.stack(forget_images)
    #         forget_labels = torch.tensor(forget_labels)
    #         forget_unique_ids = torch.tensor(forget_unique_ids)
    #
    #         retain_images = torch.stack(retain_images)
    #         retain_labels = torch.tensor(retain_labels)
    #         retain_unique_ids = torch.tensor(retain_unique_ids)
    #
    #         # Create new datasets
    #         forget_dataset = TensorDataset(forget_images, forget_labels, forget_unique_ids)
    #         retain_dataset = TensorDataset(retain_images, retain_labels, retain_unique_ids)
    #
    #         forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
    #         retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
    #
    #         hp = h_con_idx[:1000]
    #         lp = l_con_idx[-1000:]
    #         mp = m_con_idx[:1000]
    #         hp_dataset = torch.utils.data.Subset(train_loader.dataset, list(hp))
    #         mp_dataset = torch.utils.data.Subset(train_loader.dataset, list(mp))
    #         lp_dataset = torch.utils.data.Subset(train_loader.dataset, list(lp))
    #         hp_loader = replace_loader_dataset(hp_dataset, seed=seed, shuffle=True)
    #         mp_loader = replace_loader_dataset(mp_dataset, seed=seed, shuffle=True)
    #         lp_loader = replace_loader_dataset(lp_dataset, seed=seed, shuffle=True)
    #
    #     elif args.mem_proxy == 'ho_ret':
    #         loaded_heldout_retrain = np.load(f'assets/proxy_results/heldout_retrain_{args.dataset}_{args.arch}_s{seed}.npz',allow_pickle=True)
    #         loaded_klscores = loaded_heldout_retrain['kl_divergences']
    #         sorted_klscores = loaded_klscores[loaded_klscores[:, 0].argsort()]
    #         heldout_retrain_scores = sorted_klscores[:, 1]
    #
    #         indices = list(range(len(train_loader.dataset)))
    #         indices_ret = list(zip(indices, heldout_retrain_scores))
    #         indices_ret.sort(key=lambda x: x[1], reverse=True)
    #
    #         h_ret_list = indices_ret[:args.num_indexes_to_replace]
    #         l_ret_list = indices_ret[-args.num_indexes_to_replace:]
    #         h_ret_list_3000 = indices_ret[:3000]
    #         h_ret_values = [x[1] for x in h_ret_list_3000]
    #         median_ret = np.median(heldout_retrain_scores)
    #         medium_ret = (np.min(heldout_retrain_scores) + np.max(heldout_retrain_scores)) / 2
    #         medium_custom = median_ret
    #
    #         print(f'check heldout retrain: min: {np.min(heldout_retrain_scores):.3f}, max: {np.max(heldout_retrain_scores):.3f}, '
    #               f'medium: {medium_ret:.3f}, median: {median_ret:.3f}, medium_custom: {medium_custom:.3f}')
    #
    #         indices_ret.sort(key=lambda x: abs(x[1] - medium_custom))
    #         m_ret_list = indices_ret[:args.num_indexes_to_replace]
    #
    #         if args.shuffle:
    #             indices_proxy_mix = h_ret_list + l_ret_list + m_ret_list
    #             np.random.shuffle(indices_proxy_mix)
    #             h_ret_list = indices_proxy_mix[:args.num_indexes_to_replace]
    #             l_ret_list = indices_proxy_mix[-args.num_indexes_to_replace:]
    #             m_ret_list = indices_proxy_mix[args.num_indexes_to_replace:-args.num_indexes_to_replace]
    #         else:
    #             pass
    #
    #         h_ret_idx, h_ret = zip(*h_ret_list)
    #         l_ret_idx, l_ret = zip(*l_ret_list)
    #         m_ret_idx, m_ret = zip(*m_ret_list)
    #
    #         print(f'check: h_ret [{min(h_ret):.3f}, {max(h_ret):.3f}], examples: {h_ret[:10]}')
    #         print(f'check: m_ret [{min(m_ret):.3f}, {max(m_ret):.3f}], examples: {m_ret[:10]}')
    #         print(f'check: l_ret [{min(l_ret):.3f}, {max(l_ret):.3f}], examples: {l_ret[:10]}')
    #
    #         if args.mem == 'high':
    #             forget_dataset_indices = h_ret_idx
    #         elif args.mem == 'low':
    #             forget_dataset_indices = l_ret_idx
    #         elif args.mem == 'mid':
    #             forget_dataset_indices = m_ret_idx
    #         elif args.mem == 'mix':
    #             hr = h_ret_idx[:args.num_indexes_to_replace // 3]
    #             mr = m_ret_idx[:args.num_indexes_to_replace // 3]
    #             lr = l_ret_idx[-args.num_indexes_to_replace // 3:]
    #             forget_dataset_indices = hr + mr + lr
    #         else:
    #             raise ValueError('Invalid mem value')
    #
    #         forget_dataset = torch.utils.data.Subset(train_loader.dataset, list(forget_dataset_indices))
    #         all_indices = set(range(len(train_loader.dataset)))
    #         if args.sequential:
    #             if args.mem == 'low':
    #                 retain_dataset_indices = all_indices - set(l_ret_idx)
    #             elif args.mem == 'mid':
    #                 retain_dataset_indices = all_indices - set(l_ret_idx + m_ret_idx)
    #             elif args.mem == 'high':
    #                 retain_dataset_indices = all_indices - set(l_ret_idx + m_ret_idx + h_ret_idx)
    #             print('check 2, retain set size: ', len(retain_dataset_indices))
    #         else:
    #             retain_dataset_indices = all_indices - set(forget_dataset_indices)
    #             print('check 2, retain set size: ', len(retain_dataset_indices))
    #
    #         retain_dataset = torch.utils.data.Subset(train_loader.dataset, list(retain_dataset_indices))
    #         forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
    #         retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
    #
    #         hp = h_ret_idx[:1000]
    #         lp = l_ret_idx[-1000:]
    #         mp = m_ret_idx[:1000]
    #         hp_dataset = torch.utils.data.Subset(train_loader.dataset, list(hp))
    #         mp_dataset = torch.utils.data.Subset(train_loader.dataset, list(mp))
    #         lp_dataset = torch.utils.data.Subset(train_loader.dataset, list(lp))
    #         hp_loader = replace_loader_dataset(hp_dataset, seed=seed, shuffle=True)
    #         mp_loader = replace_loader_dataset(mp_dataset, seed=seed, shuffle=True)
    #         lp_loader = replace_loader_dataset(lp_dataset, seed=seed, shuffle=True)
    #
    #     else:
    #         raise ValueError('Invalid mem_proxy value')
    #
    # if fine_overlap or mem_fs_split or proxy_fs_split:
    #     forget_targets = []
    #     for _, label, _ in forget_loader.dataset:
    #         forget_targets.append(label.item())
    #
    #     unique_classes, counts = np.unique(forget_targets, return_counts=True)
    # else:
    #     print(f"number of retain dataset {len(retain_dataset)}")
    #     print(f"number of forget dataset {len(forget_dataset)}")
    #     unique_classes, counts = np.unique(forget_dataset.targets, return_counts=True)
    # class_counts = dict(zip(unique_classes.tolist(), counts.tolist()))
    # print('forget set: ')
    # print(class_counts)
    # print('retain set: ', len(retain_dataset))

    ###### end comment out for sequential unlearning ######

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
            else (0.5* (1.0+ np.cos(np.pi * ((cur_iter - args.warmup) / (args.epochs - args.warmup)))))
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
    else:
        if args.unlearn is not None:
            if args.dataset == "cifar10" or args.dataset == "TinyImagenet":
                scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
            elif args.dataset == "cifar100":
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=decreasing_lr, gamma=0.2
                )  # 0.1 is fixed
            else:
                print('scheduler not defined!')
        else:
            if args.dataset == "cifar10withids" or args.dataset == "TinyImagenetwithids":
                scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
            elif args.dataset == "cifar100withids":
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=decreasing_lr, gamma=0.2
                )  # 0.1 is fixed
            else:
                print('scheduler not defined!')

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
    # Stores the p_l, p_max and entropy values for each example
    softmax_confidences = defaultdict(list)
    p_max_values = defaultdict(list)
    p_e_values = defaultdict(list)
    acc_values = defaultdict(list)
    # Stores the averaged P_L and the step number when a learning-event occurs
    learning_events = {}

    if args.unlearn is not None:
        print("post-unlearning proxy tracking...")
        unlearn_train_loader = retain_loader
    else:
        print("pre-unlearning proxy tracking...")

    total_confidence_time = 0
    for epoch in range(start_epoch, args.epochs):
        start = time.time()
        print("Epoch #{}, Learning rate: {}".format(
                epoch, optimizer.state_dict()["param_groups"][0]["lr"]))
        logger.log({'lr': optimizer.state_dict()["param_groups"][0]["lr"]})
        # acc = train(train_loader, model, criterion, optimizer, epoch, args)
        # logger.log({"train_acc": acc})

        running_loss = 0
        correct_predictions = 0
        total_predictions = 0

        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()

        epoch_confidence_time = 0
        for i, (images, labels, unique_ids) in enumerate(train_loader):  # replace train_loader with unlearn_train_loader for post-unlearning proxy tracking
            images, labels, unique_ids = images.to(args.device), labels.to(args.device), unique_ids.to(args.device)
            model.train()
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            model.eval()

            #region:  begin comment out for tracing train_loader while training on only retain_loader
            # start_con = time.time()
            # # Calculate softmax probabilities and predictions
            # softmax_probs = torch.softmax(outputs, dim=1)
            # _, predicted = torch.max(outputs, 1)
            # total_predictions += labels.size(0)
            # correct_predictions += (predicted == labels).sum().item()

            #Track accuracy and softmax confidence for each example
            # for idx, unique_id in enumerate(unique_ids):
            #     unique_id = unique_id.item()
            #
            #     correct_class_confidence = softmax_probs[idx][labels[idx]].item()
            #     softmax_confidences[unique_id].append(correct_class_confidence)
            #
            #     p_max = softmax_probs[idx].max().item()
            #     p_max_values[unique_id].append(p_max)
            #
            #     p_e = calculate_negative_entropy(softmax_probs[idx])
            #     p_e_values[unique_id].append(p_e)
            #
            #     acc_i = 1 if predicted[idx] == labels[idx] else 0
            #     acc_values[unique_id].append(acc_i)
            #
            #     # Compute the average softmax confidence for the correct class up until this step
            #     avg_confidence = np.mean(softmax_confidences[unique_id])
            #     avg_p_max = np.mean(p_max_values[unique_id])
            #     avg_p_e = np.mean(p_e_values[unique_id])
            #     avg_acc = np.mean(acc_values[unique_id])
            #     learning_events[unique_id] = [epoch, avg_confidence, avg_p_max, avg_p_e, avg_acc]
            #endregion end comment out for tracing train_loader while training on only retain_loader

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

        ### begin comment out: this is for post-unlearning proxy tracking ###
        # model.eval()
        # for i, (images, labels, unique_ids) in enumerate(train_loader):
        #     images, labels, unique_ids = images.to(args.device), labels.to(args.device), unique_ids.to(args.device)
        #
        #     outputs = model(images)
        ### end comment out: this is for post-unlearning proxy tracking ###

            # Calculate softmax probabilities and predictions
            start_con = time.time()
            softmax_probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            # Track accuracy and softmax confidence for each example
            for idx, unique_id in enumerate(unique_ids):
                unique_id = unique_id.item()

                correct_class_confidence = softmax_probs[idx][labels[idx]].item()
                softmax_confidences[unique_id].append(correct_class_confidence)

                p_max = softmax_probs[idx].max().item()
                p_max_values[unique_id].append(p_max)

                p_e = calculate_negative_entropy(softmax_probs[idx])
                p_e_values[unique_id].append(p_e)

                acc_i = 1 if predicted[idx] == labels[idx] else 0
                acc_values[unique_id].append(acc_i)

                # Compute the average softmax confidence for the correct class up until this step
                avg_confidence = np.mean(softmax_confidences[unique_id])
                avg_p_max = np.mean(p_max_values[unique_id])
                avg_p_e = np.mean(p_e_values[unique_id])
                avg_acc = np.mean(acc_values[unique_id])
                learning_events[unique_id] = [epoch, avg_confidence, avg_p_max, avg_p_e, avg_acc]

            end_con = time.time()
            epoch_confidence_time += (end_con - start_con)


        print("train_accuracy {top1.avg:.3f}".format(top1=top1))
        print(f"Epoch {epoch} confidence time: {epoch_confidence_time:.3f}")
        total_confidence_time += epoch_confidence_time

        # evaluate on validation set
        tacc = validate_withids(val_loader, model, criterion, args)
        logger.log({"val_acc": tacc})
        # # evaluate on test set
        # test_tacc = validate(test_loader, model, criterion, args)

        scheduler.step()

        acc = top1.avg
        all_result["train_ta"].append(acc)
        all_result["val_ta"].append(tacc)
        # all_result['test_ta'].append(test_tacc)

        # remember best prec@1 and save checkpoint
        is_best_sa = tacc > best_sa
        best_sa = max(tacc, best_sa)


        """
        save_checkpoint({
            # 'state': state,
            'result': all_result,.
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
        print("one epoch duration:{}".format(time.time() - start))

    average_confidence_time = total_confidence_time / args.epochs
    print(f"Total confidence time cost: {total_confidence_time:.3f} seconds")
    print(f"Avg confidence time cost per epoch: {average_confidence_time:.3f} seconds")
    logger.log({"avg_confidence_time": average_confidence_time})
    logger.log({"total_confidence_time": total_confidence_time})

    # plot training curve
    plt.plot(all_result["train_ta"], label="train_acc")
    plt.plot(all_result["val_ta"], label="val_acc")
    # plt.plot(all_result['test_ta'], label='test_acc')
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, str(state) + "net_train.png"))
    plt.close()

    model_name = os.path.join(args.save_dir, str(state) + "model_SA_best.pth.tar")
    if args.unlearn is not None:
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
            filename='{}_original_{}_bs{}_lr{}_seed{}_epochs{}_learning_events_{}_num{}_groupid{}_proxy{}_{}.pth.tar'.format(
            args.dataset,args.arch,args.batch_size, args.lr, args.seed, args.epochs,
                args.unlearn,args.num_indexes_to_replace, args.group_index,args.mem_proxy, args.mem),
        )
    else:
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
            filename='{}_original_{}_bs{}_lr{}_seed{}_epochs{}_learning_events.pth.tar'.format(
            args.dataset,args.arch,args.batch_size, args.lr, args.seed, args.epochs),
        )

    # report result
    # check_sparsity(model)
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

    results = np.array([[k] + v for k, v in learning_events.items() if len(v) > 0])
    print("Learning events shape:", results.shape)

    metrics = dict(events=results)

    if args.unlearn is not None:
        np.savez(f'assets/proxy_results/event_results_{args.dataset}_{args.arch}_s{args.seed}_{args.unlearn}_num{args.num_indexes_to_replace}_groupid{args.group_index}_proxy{args.mem_proxy}_{args.mem}.npz', **metrics)
    else:
        np.savez(f'assets/proxy_results/event_results_{args.dataset}_{args.arch}_s{args.seed}.npz', **metrics)


if __name__ == "__main__":
    main()



