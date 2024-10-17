import copy
import os
from collections import OrderedDict
from rich import print as rich_print

import arg_parser
import evaluation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, ConcatDataset, Subset
import unlearn
import utils
from scipy.special import erf
from trainer import validate
from imagenet import get_x_y_from_data_dict
import time

from surgical_plugins.cluster import get_features, get_distance, get_fs, get_fs_dist_only


def save_gradient_ratio(data_loaders, model, criterion, args):
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.unlearn_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    gradients = {}

    forget_loader = data_loaders["forget"]
    model.eval()

    for name, param in model.named_parameters():
        gradients[name] = 0

    if args.imagenet_arch:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        for i, data in enumerate(forget_loader):
            image, target = get_x_y_from_data_dict(data, device)

            # compute output
            output_clean = model(image)
            loss = -criterion(output_clean, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradients[name] += param.grad.data

    else:
        for i, (image, target) in enumerate(forget_loader):
            image = image.cuda()
            target = target.cuda()

            # compute output
            output_clean = model(image)
            loss = -criterion(output_clean, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradients[name] += param.grad.data

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5]

    for i in threshold_list:
        sorted_dict_positions = {}
        hard_dict = {}

        # Concatenate all tensors into a single tensor
        all_elements = torch.cat([tensor.flatten() for tensor in gradients.values()])

        # Calculate the threshold index for the top 10% elements
        threshold_index = int(len(all_elements) * i)

        # Calculate positions of all elements
        positions = torch.argsort(all_elements)
        ranks = torch.argsort(positions)

        start_index = 0
        for key, tensor in gradients.items():
            num_elements = tensor.numel()
            # tensor_positions = positions[start_index: start_index + num_elements]
            tensor_ranks = ranks[start_index : start_index + num_elements]

            sorted_positions = tensor_ranks.reshape(tensor.shape)
            sorted_dict_positions[key] = sorted_positions

            # Set the corresponding elements to 1
            threshold_tensor = torch.zeros_like(tensor_ranks)
            threshold_tensor[tensor_ranks < threshold_index] = 1
            threshold_tensor = threshold_tensor.reshape(tensor.shape)
            hard_dict[key] = threshold_tensor
            start_index += num_elements

        all_gradients = torch.cat(
            [gradient.flatten() for gradient in gradients.values()]
        )

        sigmoid_gradients = torch.abs(2 * (torch.sigmoid(all_gradients) - 0.5))
        tanh_gradients = torch.abs(torch.tanh(all_gradients))

        sigmoid_soft_dict = {}
        tanh_soft_dict = {}
        start_idx = 0
        for net_name, gradient in gradients.items():
            num_params = gradient.numel()
            end_idx = start_idx + num_params
            sigmoid_gradient = sigmoid_gradients[start_idx:end_idx]
            sigmoid_gradient = sigmoid_gradient.reshape(gradient.shape)
            sigmoid_soft_dict[net_name] = sigmoid_gradient

            tanh_gradient = tanh_gradients[start_idx:end_idx]
            tanh_gradient = tanh_gradient.reshape(gradient.shape)
            tanh_soft_dict[net_name] = tanh_gradient
            start_idx = end_idx

        if args.mem_proxy is not None:
            torch.save(
                sigmoid_soft_dict,
                os.path.join(args.save_dir, "sigmoid_soft_{}_{}_{}_{}_num{}_groupid{}_proxy{}_{}_seed{}.pt".format(
                    i, args.dataset, args.arch, args.class_to_replace, args.num_indexes_to_replace,
                args.group_index, args.mem_proxy, args.mem, args.seed)),
            )
            torch.save(
                tanh_soft_dict, os.path.join(args.save_dir, "tanh_soft_{}_{}_{}_{}_num{}_groupid{}_proxy{}_{}_seed{}.pt".format(
                    i, args.dataset, args.arch, args.class_to_replace, args.num_indexes_to_replace,
                args.group_index, args.mem_proxy, args.mem, args.seed))
            )
            torch.save(hard_dict, os.path.join(args.save_dir, "hard_{}_{}_{}_{}_num{}_groupid{}_proxy{}_{}_seed{}.pt".format(
                i, args.dataset, args.arch, args.class_to_replace, args.num_indexes_to_replace,
                args.group_index, args.mem_proxy, args.mem, args.seed))
            )
        else:
            torch.save(
                sigmoid_soft_dict,
                os.path.join(args.save_dir, "sigmoid_soft_{}_{}_{}_{}_num{}_groupid{}_mem{}_seed{}.pt".format(
                    i ,args.dataset, args.arch, args.class_to_replace, args.num_indexes_to_replace,
                args.group_index, args.mem, args.seed)),
            )
            torch.save(
                tanh_soft_dict, os.path.join(args.save_dir, "tanh_soft_{}_{}_{}_{}_num{}_groupid{}_mem{}_seed{}.pt".format(
                    i ,args.dataset, args.arch, args.class_to_replace, args.num_indexes_to_replace,
                args.group_index, args.mem, args.seed))
            )
            torch.save(hard_dict, os.path.join(args.save_dir, "hard_{}_{}_{}_{}_num{}_groupid{}_mem{}_seed{}.pt".format(
                i ,args.dataset, args.arch, args.class_to_replace, args.num_indexes_to_replace,
                args.group_index, args.mem, args.seed)))


def load_pth_tar_files(folder_path):
    pth_tar_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pt"):
                file_path = os.path.join(root, file)
                pth_tar_files.append(file_path)

    return pth_tar_files


def compute_gradient_ratio(mask_path):
    mask = torch.load(mask_path)
    all_elements = torch.cat([tensor.flatten() for tensor in mask.values()])
    ones_tensor = torch.ones(all_elements.shape)
    ratio = torch.sum(all_elements) / torch.sum(ones_tensor)
    name = mask_path.split("/")[-1].replace(".pt", "")
    return name, ratio


def print_gradient_ratio(mask_folder, save_path):
    ratio_dict = {}
    mask_path_list = load_pth_tar_files(mask_folder)
    for i in mask_path_list:
        name, ratio = compute_gradient_ratio(i)
        print(name, ratio)
        ratio_dict[name] = ratio.item()

    ratio_df = pd.DataFrame([ratio_dict])
    ratio_df.to_csv(save_path + "ratio_df.csv", index=False)


def main():
    start_rte = time.time()
    args = arg_parser.parse_args()
    # print(args.choice, type(args.choice), len(args.choice))

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed

    args.mask = 'assets/checkpoints/0{}_original_{}_bs256_lr0.1_seed{}_epochs{}.pth.tar'.format(
                args.dataset, args.arch, args.seed, args.epochs)

    # prepare dataset
    if args.dataset == "cifar10" or args.dataset == "cifar100":
        (
            model,
            train_loader_full,
            val_loader,
            test_loader,
            marked_loader,
            train_idx
        ) = utils.setup_model_dataset(args)
    elif args.dataset == "TinyImagenet":
        args.data_dir = "/data/image_data/tiny-imagenet-200/"
        (
            model,
            train_loader_full,
            val_loader,
            test_loader,
            marked_loader,
        ) = utils.setup_model_dataset(args)
    model.cuda()
    rich_print(args)

    def replace_loader_dataset(
        dataset, batch_size=args.batch_size, seed=1, shuffle=True
    ):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=shuffle,
        )

    forget_dataset = copy.deepcopy(marked_loader.dataset)

    if args.mem is not None and args.group_index is None and args.mem_proxy is None:
        fine_overlap = False
        mem_fs_split = True
        proxy_fs_split = False
    elif args.mem is None and args.group_index is not None and args.mem_proxy is None:
        fine_overlap = True
        mem_fs_split = False
        proxy_fs_split = False
    elif args.mem_proxy is not None:
        fine_overlap = False
        mem_fs_split = False
        proxy_fs_split = True
    else:
        fine_overlap = False
        mem_fs_split = False
        proxy_fs_split = False

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
    elif fine_overlap:
        train_loader = DataLoader(
            train_loader_full.dataset,
            batch_size=args.batch_size,
            shuffle=False
        )

        index_mapping = {i: data for i, data in enumerate(train_loader.dataset)}

        print('Loading original model...')
        filename = '0{}_original_{}_bs{}_lr{}_seed{}_epochs{}.pth.tar'.format(
            args.dataset, args.arch, args.batch_size, args.lr, args.seed, args.epochs)
        pruning = state = 0
        model_og = copy.deepcopy(model)
        checkpoint = utils.load_checkpoint(device, 'assets/checkpoints', state, filename)
        model_og.load_state_dict(checkpoint['state_dict'])
        model_og.eval()

        if args.dataset == 'cifar10':
            num_clusters = 10
            n_group = 15
        elif args.dataset == 'cifar100':
            num_clusters = 100
            n_group = 15
        elif args.dataset == 'TinyImagenet':
            num_clusters = 200
            n_group = 100
        distances_matrix, _, _ = get_distance(train_loader, model_og, args, cluster_state=False, num_clusters=num_clusters,
                                              vis_cluster=False, vis_distribution=False)

        n_sample = args.num_indexes_to_replace
        print('Group index: ', args.group_index)
        if args.dataset == 'cifar10' or args.dataset == 'cifar100':
            _, _, l_des_idx, _ = get_fs_dist_only(distances_matrix, train_loader, n_group=n_group, n_sample=n_sample, group_index=3)
            _, _, m_des_idx, _ = get_fs_dist_only(distances_matrix, train_loader, n_group=n_group, n_sample=n_sample, group_index=12)
            _, _, h_des_idx, _ = get_fs_dist_only(distances_matrix, train_loader, n_group=n_group, n_sample=n_sample, group_index=14)
        elif args.dataset == 'TinyImagenet':
            _, _, l_des_idx_1, _ = get_fs_dist_only(distances_matrix, train_loader, n_group=n_group, n_sample=n_sample, group_index=0)
            _, _, l_des_idx_2, _ = get_fs_dist_only(distances_matrix, train_loader, n_group=n_group, n_sample=n_sample, group_index=1)
            _, _, l_des_idx_3, _ = get_fs_dist_only(distances_matrix, train_loader, n_group=n_group, n_sample=n_sample, group_index=2)
            _, _, m_des_idx_1, _ = get_fs_dist_only(distances_matrix, train_loader, n_group=n_group, n_sample=n_sample, group_index=49)
            _, _, m_des_idx_2, _ = get_fs_dist_only(distances_matrix, train_loader, n_group=n_group, n_sample=n_sample, group_index=50)
            _, _, m_des_idx_3, _ = get_fs_dist_only(distances_matrix, train_loader, n_group=n_group, n_sample=n_sample, group_index=51)
            _, _, h_des_idx_1, _ = get_fs_dist_only(distances_matrix, train_loader, n_group=n_group, n_sample=n_sample, group_index=97)
            _, _, h_des_idx_2, _ = get_fs_dist_only(distances_matrix, train_loader, n_group=n_group, n_sample=n_sample, group_index=98)
            _, _, h_des_idx_3, _ = get_fs_dist_only(distances_matrix, train_loader, n_group=n_group, n_sample=n_sample, group_index=99)
            l_des_idx = l_des_idx_1 + l_des_idx_2 + l_des_idx_3
            m_des_idx = m_des_idx_1 + m_des_idx_2 + m_des_idx_3
            h_des_idx = h_des_idx_1 + h_des_idx_2 + h_des_idx_3


        print('check: Group index: ', args.group_index)
        if args.group_index == 3:
            forget_dataset_indices = l_des_idx
        elif args.group_index == 12:
            forget_dataset_indices = m_des_idx
        elif args.group_index == 14:
            forget_dataset_indices = h_des_idx
        elif args.group_index == -1:
            ld = l_des_idx[:args.num_indexes_to_replace // 3]
            md = m_des_idx[:args.num_indexes_to_replace // 3]
            hd = h_des_idx[:args.num_indexes_to_replace // 3]
            forget_dataset_indices = ld + md + hd
        else:
            raise ValueError('Invalid mem value')

        forget_dataset = torch.utils.data.Subset(train_loader.dataset, list(forget_dataset_indices))
        all_indices = set(range(len(train_loader.dataset)))
        if args.sequential:
            retain_dataset_indices = all_indices - set(l_mem_idx + h_mem_idx + m_mem_idx)
        else:
            retain_dataset_indices = all_indices - set(forget_dataset_indices)

        retain_dataset = torch.utils.data.Subset(train_loader.dataset, list(retain_dataset_indices))
        forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
        retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)

        hd = list(h_des_idx)[:1000]
        md = list(m_des_idx)[:1000]
        ld = list(l_des_idx)[:1000]
        hd_dataset = torch.utils.data.Subset(train_loader.dataset, list(hd))
        md_dataset = torch.utils.data.Subset(train_loader.dataset, list(md))
        ld_dataset = torch.utils.data.Subset(train_loader.dataset, list(ld))
        hd_loader = replace_loader_dataset(hd_dataset, seed=seed, shuffle=True)
        md_loader = replace_loader_dataset(md_dataset, seed=seed, shuffle=True)
        ld_loader = replace_loader_dataset(ld_dataset, seed=seed, shuffle=True)
    elif mem_fs_split:
        print('[fs split]: mem_fs_split')
        train_loader = DataLoader(
            train_loader_full.dataset,
            batch_size=args.batch_size,
            shuffle=False
        )

        if args.dataset == 'cifar10':
            loaded_results = np.load('estimates_results_woShuffle.npz')
            loaded_memorization = loaded_results['memorization']
        elif args.dataset == 'cifar100':
            loaded_results = np.load('cifar100_infl_matrix.npz')
            loaded_memorization = loaded_results['tr_mem']
        loaded_memorization = loaded_memorization[train_idx]

        indices = list(range(len(train_loader.dataset)))
        indices_mem = list(zip(indices, loaded_memorization))
        np.random.shuffle(indices_mem)

        indices_mem.sort(key=lambda x: x[1], reverse=True)  # sort by memorization in descending order
        h_mem_list = indices_mem[:args.num_indexes_to_replace]
        l_mem_list = indices_mem[-args.num_indexes_to_replace:]
        indices_mem.sort(key=lambda x: abs(x[1] - 0.5))
        m_mem_list = indices_mem[:args.num_indexes_to_replace]

        if args.shuffle:
            indices_mem_mix = h_mem_list + l_mem_list + m_mem_list
            np.random.shuffle(indices_mem_mix)
            h_mem_list = indices_mem_mix[:args.num_indexes_to_replace]
            l_mem_list = indices_mem_mix[-args.num_indexes_to_replace:]
            m_mem_list = indices_mem_mix[args.num_indexes_to_replace:-args.num_indexes_to_replace]
        else:
            pass

        h_mem_idx, h_mem = zip(*h_mem_list)
        l_mem_idx, l_mem = zip(*l_mem_list)
        m_mem_idx, m_mem = zip(*m_mem_list)

        print('check: args.mem: ', args.mem)
        if args.mem == 'high':
            forget_dataset_indices = h_mem_idx
        elif args.mem == 'low':
            forget_dataset_indices = l_mem_idx
        elif args.mem == 'mid':
            forget_dataset_indices = m_mem_idx
        elif args.mem == 'mix':
            hm = h_mem_idx[:args.num_indexes_to_replace // 3]
            mm = m_mem_idx[:args.num_indexes_to_replace // 3]
            lm = l_mem_idx[-args.num_indexes_to_replace // 3:]
            forget_dataset_indices = hm + lm + mm
        else:
            raise ValueError('Invalid mem value')

        forget_dataset = torch.utils.data.Subset(train_loader.dataset, list(forget_dataset_indices))
        all_indices = set(range(len(train_loader.dataset)))
        if args.sequential:
            if args.mem == 'low':
                retain_dataset_indices = all_indices - set(l_mem_idx)
            elif args.mem == 'mid':
                retain_dataset_indices = all_indices - set(l_mem_idx + m_mem_idx)
            elif args.mem == 'high':
                retain_dataset_indices = all_indices - set(l_mem_idx + m_mem_idx + h_mem_idx)
            print('check 2, retain set size: ', len(retain_dataset_indices))
        else:
            retain_dataset_indices = all_indices - set(forget_dataset_indices)
            print('check 2, retain set size: ', len(retain_dataset_indices))

        retain_dataset = torch.utils.data.Subset(train_loader.dataset, list(retain_dataset_indices))
        forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
        retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)

        hm = h_mem_idx[:1000]
        lm = l_mem_idx[:1000]
        mm = m_mem_idx[:1000]
        hm_dataset = torch.utils.data.Subset(train_loader.dataset, list(hm))
        mm_dataset = torch.utils.data.Subset(train_loader.dataset, list(mm))
        lm_dataset = torch.utils.data.Subset(train_loader.dataset, list(lm))
        hm_loader = replace_loader_dataset(hm_dataset, seed=seed, shuffle=True)
        mm_loader = replace_loader_dataset(mm_dataset, seed=seed, shuffle=True)
        lm_loader = replace_loader_dataset(lm_dataset, seed=seed, shuffle=True)
    elif proxy_fs_split:
        print(f'[fs split]: proxy of memorization - {args.mem_proxy}')
        train_loader = DataLoader(
            train_loader_full.dataset,
            batch_size=args.batch_size,
            shuffle=False
        )
        if args.mem_proxy == 'confidence' or args.mem_proxy == 'max_conf' or args.mem_proxy == 'entropy' or args.mem_proxy == 'bi_acc':
            loaded_results = np.load(f'assets/proxy_results/event_results_{args.dataset}withids_{args.arch}_s{seed}.npz',
                                     allow_pickle=True)
            loaded_events = loaded_results['events']
            sorted_events = loaded_events[loaded_events[:, 0].argsort()]

            nan_or_inf_indices = np.isnan(loaded_events) | np.isinf(loaded_events)
            print("Number of NaNs or infinities:", np.sum(nan_or_inf_indices))
            sorted_events = np.nan_to_num(sorted_events, nan=np.inf)

            if args.mem_proxy == 'confidence':
                sorted_confidences = sorted_events[:, 2]
            elif args.mem_proxy == 'max_conf':
                sorted_confidences = sorted_events[:, 3]
            elif args.mem_proxy == 'entropy':
                sorted_confidences = sorted_events[:, 4]
            elif args.mem_proxy == 'bi_acc':
                sorted_confidences = sorted_events[:, 5]

            indices = list(range(len(train_loader.dataset)))
            indices_con = list(zip(indices, sorted_confidences))
            indices_con.sort(key=lambda x: x[1], reverse=True)

            if args.mem_proxy == 'entropy':
                indices_con = [x for x in indices_con if not np.isinf(x[1])]
                print('check: number of values left after removing nan or inf: ', len(indices_con))

            h_con_list = indices_con[:args.num_indexes_to_replace]
            l_con_list = indices_con[-args.num_indexes_to_replace:]
            # get the median confidence value
            median_con = np.median(sorted_confidences)
            medium_con = (np.min(sorted_confidences) + np.max(sorted_confidences)) / 2
            print('check: min confidence: ', np.min(sorted_confidences), 'max confidence: ', np.max(sorted_confidences), 'medium confidence: ', medium_con, 'median confidence: ', median_con)
            indices_con.sort(key=lambda x: abs(x[1] - medium_con))
            m_con_list = indices_con[:args.num_indexes_to_replace]

            if args.shuffle:
                indices_proxy_mix = h_con_list + l_con_list + m_con_list
                np.random.shuffle(indices_proxy_mix)
                h_con_list = indices_proxy_mix[:args.num_indexes_to_replace]
                l_con_list = indices_proxy_mix[-args.num_indexes_to_replace:]
                m_con_list = indices_proxy_mix[args.num_indexes_to_replace:-args.num_indexes_to_replace]

            else:
                pass

            h_con_idx, h_con = zip(*h_con_list)
            l_con_idx, l_con = zip(*l_con_list)
            m_con_idx, m_con = zip(*m_con_list)

            print(f'check: h_con [{min(h_con):.3f}, {max(h_con):.3f}], examples: {h_con[:10]}')
            print(f'check: m_con [{min(m_con):.3f}, {max(m_con):.3f}], examples: {m_con[:10]}')
            print(f'check: l_con [{min(l_con):.3f}, {max(l_con):.3f}], examples: {l_con[:10]}')

            if args.mem == 'high':
                forget_dataset_indices = h_con_idx
            elif args.mem == 'low':
                forget_dataset_indices = l_con_idx
            elif args.mem == 'mid':
                forget_dataset_indices = m_con_idx
            elif args.mem == 'mix':
                hc = h_con_idx[:args.num_indexes_to_replace // 3]
                mc = m_con_idx[:args.num_indexes_to_replace // 3]
                lc = l_con_idx[-args.num_indexes_to_replace // 3:]
                forget_dataset_indices = hc + mc + lc
            else:
                raise ValueError('Invalid mem value')

            forget_dataset = torch.utils.data.Subset(train_loader.dataset, list(forget_dataset_indices))
            all_indices = set(range(len(train_loader.dataset)))
            if args.sequential:
                if args.mem == 'high':
                    retain_dataset_indices = all_indices - set(h_con_idx)
                elif args.mem == 'mid':
                    retain_dataset_indices = all_indices - set(h_con_idx + m_con_idx)
                elif args.mem == 'low':
                    retain_dataset_indices = all_indices - set(h_con_idx + m_con_idx + l_con_idx)
                print('check 2, retain set size: ', len(retain_dataset_indices))
            else:
                retain_dataset_indices = all_indices - set(forget_dataset_indices)
                print('check 2, retain set size: ', len(retain_dataset_indices))

            retain_dataset = torch.utils.data.Subset(train_loader.dataset, list(retain_dataset_indices))
            forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
            retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)

            hp = h_con_idx[:1000]
            lp = l_con_idx[-1000:]
            mp = m_con_idx[:1000]
            hp_dataset = torch.utils.data.Subset(train_loader.dataset, list(hp))
            mp_dataset = torch.utils.data.Subset(train_loader.dataset, list(mp))
            lp_dataset = torch.utils.data.Subset(train_loader.dataset, list(lp))
            hp_loader = replace_loader_dataset(hp_dataset, seed=seed, shuffle=True)
            mp_loader = replace_loader_dataset(mp_dataset, seed=seed, shuffle=True)
            lp_loader = replace_loader_dataset(lp_dataset, seed=seed, shuffle=True)

        elif args.mem_proxy == 'ho_ret':
            loaded_heldout_retrain = np.load(f'assets/proxy_results/heldout_retrain_{args.dataset}_{args.arch}_s{seed}.npz',allow_pickle=True)
            loaded_klscores = loaded_heldout_retrain['kl_divergences']
            sorted_klscores = loaded_klscores[loaded_klscores[:, 0].argsort()]
            heldout_retrain_scores = sorted_klscores[:, 1]

            indices = list(range(len(train_loader.dataset)))
            indices_ret = list(zip(indices, heldout_retrain_scores))
            indices_ret.sort(key=lambda x: x[1], reverse=True)

            h_ret_list = indices_ret[:args.num_indexes_to_replace]
            l_ret_list = indices_ret[-args.num_indexes_to_replace:]
            # get the median heldout retrain value
            h_ret_list_3000 = indices_ret[:3000]
            h_ret_values = [x[1] for x in h_ret_list_3000]
            median_ret = np.median(heldout_retrain_scores)
            medium_ret = (np.min(heldout_retrain_scores) + np.max(heldout_retrain_scores)) / 2
            medium_custom = median_ret

            print(f'check heldout retrain: min: {np.min(heldout_retrain_scores):.3f}, max: {np.max(heldout_retrain_scores):.3f}, '
                  f'medium: {medium_ret:.3f}, median: {median_ret:.3f}, medium_custom: {medium_custom:.3f}')

            indices_ret.sort(key=lambda x: abs(x[1] - medium_custom))
            m_ret_list = indices_ret[:args.num_indexes_to_replace]

            if args.shuffle:
                indices_proxy_mix = h_ret_list + l_ret_list + m_ret_list
                np.random.shuffle(indices_proxy_mix)
                h_ret_list = indices_proxy_mix[:args.num_indexes_to_replace]
                l_ret_list = indices_proxy_mix[-args.num_indexes_to_replace:]
                m_ret_list = indices_proxy_mix[args.num_indexes_to_replace:-args.num_indexes_to_replace]
            else:
                pass

            h_ret_idx, h_ret = zip(*h_ret_list)
            l_ret_idx, l_ret = zip(*l_ret_list)
            m_ret_idx, m_ret = zip(*m_ret_list)

            print(f'check: h_ret [{min(h_ret):.3f}, {max(h_ret):.3f}], examples: {h_ret[:10]}')
            print(f'check: m_ret [{min(m_ret):.3f}, {max(m_ret):.3f}], examples: {m_ret[:10]}')
            print(f'check: l_ret [{min(l_ret):.3f}, {max(l_ret):.3f}], examples: {l_ret[:10]}')

            if args.mem == 'high':
                forget_dataset_indices = h_ret_idx
            elif args.mem == 'low':
                forget_dataset_indices = l_ret_idx
            elif args.mem == 'mid':
                forget_dataset_indices = m_ret_idx
            elif args.mem == 'mix':
                hr = h_ret_idx[:args.num_indexes_to_replace // 3]
                mr = m_ret_idx[:args.num_indexes_to_replace // 3]
                lr = l_ret_idx[-args.num_indexes_to_replace // 3:]
                forget_dataset_indices = hr + mr + lr
            else:
                raise ValueError('Invalid mem value')

            forget_dataset = torch.utils.data.Subset(train_loader.dataset, list(forget_dataset_indices))
            all_indices = set(range(len(train_loader.dataset)))
            if args.sequential:
                if args.mem == 'low':
                    retain_dataset_indices = all_indices - set(l_ret_idx)
                elif args.mem == 'mid':
                    retain_dataset_indices = all_indices - set(l_ret_idx + m_ret_idx)
                elif args.mem == 'high':
                    retain_dataset_indices = all_indices - set(l_ret_idx + m_ret_idx + h_ret_idx)
                print('check 2, retain set size: ', len(retain_dataset_indices))
            else:
                retain_dataset_indices = all_indices - set(forget_dataset_indices)
                print('check 2, retain set size: ', len(retain_dataset_indices))

            retain_dataset = torch.utils.data.Subset(train_loader.dataset, list(retain_dataset_indices))
            forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
            retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)

            hp = h_ret_idx[:1000]
            lp = l_ret_idx[-1000:]
            mp = m_ret_idx[:1000]
            hp_dataset = torch.utils.data.Subset(train_loader.dataset, list(hp))
            mp_dataset = torch.utils.data.Subset(train_loader.dataset, list(mp))
            lp_dataset = torch.utils.data.Subset(train_loader.dataset, list(lp))
            hp_loader = replace_loader_dataset(hp_dataset, seed=seed, shuffle=True)
            mp_loader = replace_loader_dataset(mp_dataset, seed=seed, shuffle=True)
            lp_loader = replace_loader_dataset(lp_dataset, seed=seed, shuffle=True)

        else:
            raise ValueError('Invalid mem_proxy value')

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

    if fine_overlap or mem_fs_split or proxy_fs_split:
        forget_targets = [train_loader.dataset.targets[i] for i in forget_dataset.indices]
        unique_classes, counts = np.unique(forget_targets, return_counts=True)
    else:
        print(f"number of retain dataset {len(retain_dataset)}")
        print(f"number of forget dataset {len(forget_dataset)}")
        unique_classes, counts = np.unique(forget_dataset.targets, return_counts=True)
    class_counts = dict(zip(unique_classes.tolist(), counts.tolist()))
    print('forget set: ')
    print(class_counts)
    print('retain set: ', len(retain_dataset))


    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )

    criterion = nn.CrossEntropyLoss()

    evaluation_result = None

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        checkpoint = torch.load(args.mask, map_location=device)
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]

        if args.unlearn != "retrain":
            model.load_state_dict(checkpoint, strict=False)

        start_unlearn = time.time()
        save_gradient_ratio(unlearn_data_loaders, model, criterion, args)

    end_rte = time.time()
    print(f'Overall time taken for unlearning & preparation: {end_rte - start_rte:.3f}s')
    print(f'Time taken for generating mask only: {end_rte - start_unlearn:.3f}s')

if __name__ == "__main__":
    main()
