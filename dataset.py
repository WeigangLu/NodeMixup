#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch
import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Coauthor, Reddit2
from torch_geometric.utils import to_networkx, degree, add_self_loops
import networkx as nx
import numpy as np
import os
import argparse
from torch_geometric.utils.num_nodes import maybe_num_nodes
from typing import Optional, Tuple, Union
from torch import Tensor


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_splits_CS(labels, num_classes, percls_trn=20, percls_val=30):
    num_nodes = labels.shape[0]
    indices = []
    for i in range(num_classes):
        index = (labels == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)
    val_index = torch.cat([i[percls_trn:percls_trn + percls_val] for i in indices], dim=0)

    rest_index = torch.cat([i[percls_trn + percls_val:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    train_mask = index_to_mask(train_index, size=num_nodes)
    val_mask = index_to_mask(val_index, size=num_nodes)
    test_mask = index_to_mask(rest_index, size=num_nodes)

    return train_mask, val_mask, test_mask


def random_splits(labels, num_classes, percls_trn=20, val_size=500, test_size=1000):
    num_nodes = labels.shape[0]
    indices = []
    for i in range(num_classes):
        index = (labels == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    train_mask = index_to_mask(train_index, size=num_nodes)
    val_mask = index_to_mask(rest_index[:val_size], size=num_nodes)
    test_mask = index_to_mask(rest_index[val_size:val_size + test_size], size=num_nodes)

    return train_mask, val_mask, test_mask


def DataLoader(name, args):
    dataset = None
    name = name.lower()
    root_path = args.root_path
    path = osp.join(root_path, 'data')
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, name, transform=T.NormalizeFeatures())
    elif name in ["coauthor", "physics"]:
        name = "CS" if name != "physics" else "physics"
        dataset = Coauthor(path, name, transform=T.NormalizeFeatures())
    elif name in ['ogbn-arxiv']:
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name=name, transform=T.ToSparseTensor())
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')

    return dataset


def load_data(args):
    dname = args.dataset
    dataset = DataLoader(dname, args)

    return dataset
