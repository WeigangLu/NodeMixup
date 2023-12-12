#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# Distributed under terms of the MIT license.


from dataset import random_splits, random_splits_CS, index_to_mask
from model import BaselineModels
import torch
import random

import time
from utils import Cross_entropy_loss, Accuracy, sharpen
import numpy as np
from tqdm import tqdm
from mixer import Mixer


def seed_setting(seed):
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    else:
        pass


def train(model, optimizer, x, adj, mask_dict, labels, mixup_dict=None, use_mixup=False):
    model.train()

    optimizer.zero_grad()
    logits, eq_mixup_logits, neq_mixup_logits = model(x, adj, mixup_dict)
    loss = Cross_entropy_loss(logits, labels, mask_dict['train'])
    if use_mixup:
        mixer = mixup_dict['mixer']
        eq_mixup_loss, neq_mixup_loss = mixer.mixup_loss(eq_mixup_logits, neq_mixup_logits, mixup_dict)
        mixup_loss = mixup_dict['lam_intra'] * eq_mixup_loss + mixup_dict['lam_inter'] * neq_mixup_loss
        # print(f"eq_mixup loss:{eq_mixup_loss:.4f}, neq_mixup loss:{neq_mixup_loss:.4f}")
        loss += mixup_loss

    loss.backward()
    optimizer.step()

    del logits


@torch.no_grad()
def test(model, x, adj, args, mask_dict, labels, mixup_dict):
    model.eval()
    logits, eq_mixup_logits, neq_mixup_logits = model(x, adj, mixup_dict)
    accs = []
    losses = []
    for mask in (mask_dict['train'], mask_dict['valid'], mask_dict['test']):
        acc, false = Accuracy(logits, labels, mask)
        loss = Cross_entropy_loss(logits, labels, mask)
        accs.append(acc)
        losses.append(loss.detach().cpu().item())

    return accs, losses, logits


def data_split(dataset, data, train_size, args, dataloader=None):
    mask_dict = dict()
    mask_dict['train'] = None
    mask_dict['valid'] = None
    mask_dict['test'] = None
    rc = None
    if dataset in ('cora', 'citeseer', 'pubmed', "reddit"):
        if train_size > 0:

            mask_dict['train'], mask_dict['valid'], mask_dict['test'] = random_splits(labels=data.y,
                                                                                      num_classes=max(data.y) + 1,
                                                                                      percls_trn=train_size,
                                                                                      val_size=data.val_mask.sum(),
                                                                                      test_size=data.test_mask.sum())
        else:

            mask_dict['train'], mask_dict['valid'], mask_dict['test'] = (data.train_mask, data.val_mask, data.test_mask)
    elif dataset in ('Coauthor', 'Physics'):
        if train_size < 0:
            mask_dict['train'], mask_dict['valid'], mask_dict['test'] = random_splits_CS(labels=data.y,
                                                                                         num_classes=max(data.y) + 1,
                                                                                         percls_trn=20,
                                                                                         percls_val=30)
        else:
            mask_dict['train'], mask_dict['valid'], mask_dict['test'] = random_splits_CS(labels=data.y,
                                                                                         num_classes=max(data.y) + 1,
                                                                                         percls_trn=train_size,
                                                                                         percls_val=30)
    elif dataset == 'ogbn-arxiv':
        split_idx = dataloader.get_idx_split()
        data.edge_index = data.adj_t.to_symmetric()
        data.y = data.y.squeeze(1)
        if train_size > 0:
            mask_dict['train'], mask_dict['valid'], mask_dict['test'] = random_splits(labels=data.y,
                                                                                      num_classes=dataloader.num_classes,
                                                                                      percls_trn=train_size,
                                                                                      val_size=split_idx['valid'].shape[
                                                                                          0],
                                                                                      test_size=split_idx['test'].shape[
                                                                                          0])
        else:
            mask_dict['train'], mask_dict['valid'], mask_dict['test'] = (
                split_idx['train'], split_idx['valid'], split_idx['test'])
            mask_dict['train'] = index_to_mask(mask_dict['train'], data.y.shape[0])
            mask_dict['valid'] = index_to_mask(mask_dict['valid'], data.y.shape[0])
            mask_dict['test'] = index_to_mask(mask_dict['test'], data.y.shape[0])

    mask_dict['unlabeled'] = (1 - mask_dict['train'].float()).bool()

    return mask_dict, data.y, rc


def Trail(args, data, model, dataloader=None):
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    mask_dict, labels, rc = data_split(dataset=args.dataset, data=data, train_size=args.train_size, args=args,
                                       dataloader=dataloader)

    model, data, labels = model.to(device), data.to(device), labels.to(device)
    # calculate parameters
    num_paras = 0
    for p in model.parameters():
        num_paras += p.numel()
    print('# Parameters = {}\n'.format(num_paras))

    optimizer = torch.optim.Adam([
        {
            'params': model.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },

    ],
        lr=args.lr)

    result_dict = dict()
    var_dict = dict()
    var_dict['rc'] = rc
    var_dict['test_mask'] = mask_dict['test']
    var_dict['train_mask'] = mask_dict['train']
    var_dict['labels'] = labels
    var_dict['num_paras'] = num_paras

    strategy_dict = dict()

    use_mixup = args.lam_intra != 0 or args.lam_inter != 0

    mixup_dict = None
    if use_mixup:
        nclass = labels.max() + 1
        t_idx = torch.nonzero(mask_dict['train']).squeeze(-1).to(device)
        t_labels = labels[mask_dict['train']].unsqueeze(1).to(device)
        t_y = torch.zeros(t_idx.shape[0], nclass, device=device).scatter_(1, t_labels, 1).to(device)
        un_idx = torch.nonzero(mask_dict['unlabeled']).squeeze(-1).to(device)
        mixup_y = torch.zeros(labels.shape[0], nclass).to(device)
        mixup_y[un_idx] = 1. / nclass

        mixer = Mixer(t_idx, un_idx, beta_d=args.beta_d, beta_s=args.beta_s, temp=args.temp,
                      train_size=int(t_idx.shape[0] / nclass), nclass=nclass, alpha=args.mixup_alpha,
                      gamma=args.gamma, device=device)

        mixup_dict = dict()
        mixup_dict['t_idx'], mixup_dict['t_y'], mixup_dict['un_idx'], mixup_dict['all_idx'], mixup_dict['mixer'] = \
            t_idx, t_y, un_idx, torch.cat([t_idx, un_idx]), mixer
        mixup_dict['lam_intra'], mixup_dict['lam_inter'] = args.lam_intra, args.lam_inter

    start = time.time()

    for epoch in tqdm(range(args.epochs)):
        x = data.x
        adj = data.edge_index
        if use_mixup:
            mixup_y[t_idx] = t_y
            mixup_dict['eq_mixup_x'], mixup_dict['eq_mixup_y'], mixup_dict['neq_mixup_x'], mixup_dict['neq_mixup_y'], \
            mixup_dict['mixup_adj'], mixup_dict['E'], mixup_dict['eq_idx'] = \
                mixer.mixup_data(x, mixup_y, adj)
        train(model, optimizer, x, adj, labels=labels, mask_dict=mask_dict, mixup_dict=mixup_dict,
              use_mixup=use_mixup)

        result_dict['epoch'] = epoch
        [result_dict['train_acc'], result_dict['val_acc'], result_dict['test_acc']], \
        [result_dict['train_loss'], result_dict['val_loss'], result_dict['test_loss']], \
        logits = test(model, x, adj, args, labels=labels, mask_dict=mask_dict, mixup_dict=mixup_dict)
        if use_mixup:
            y = logits.softmax(-1).detach()
            y_val, _ = torch.max(y, 1)
            mask = y_val >= args.gamma
            mask = mask.cpu()
            if mask[mask_dict['unlabeled']].sum() > 0:
                target_y = y[mask * mask_dict['unlabeled']]
                mixup_y[mask * mask_dict['unlabeled']] = sharpen(target_y, 0.5)
        yield result_dict, var_dict

    end = time.time()
    print(f"time cost:{(end - start) / args.epochs:.2f} per epoch")


def trail_run(args, dataset):
    seed_setting(args.seed)
    args.out_dim = dataset.num_classes
    data = dataset[0]
    print('%d train samples per class on %s' % (args.train_size, args.dataset))
    model = BaselineModels(dataset=dataset, model=args.model, args=args)

    return Trail(args=args, data=data, model=model, dataloader=dataset)
