import argparse
from pickle import TRUE
import sys
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops, subgraph, dropout_adj
from torch_scatter import scatter
import copy

from logger import Logger, SimpleLogger
from dataset import load_dataset
from data_utils import normalize, gen_normalized_adjs, evaluate, eval_acc, eval_rocauc, eval_f1, to_sparse_tensor, \
    load_fixed_splits, adj_mul, get_gpu_memory_map, count_parameters
from parse import parse_method, parser_add_main_args

import time
import warnings
warnings.filterwarnings('ignore')

from pmlp import *

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset = load_dataset(args.data_dir, args.dataset, args.sub_dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

if args.rand_split:
    split_idx_lst = dataset.get_idx_split(split_type='random_all', train_prop=args.train_prop, valid_prop=args.valid_prop)
elif args.rand_split_class:
    split_idx_lst = dataset.get_idx_split(split_type='random_class', tr_num_per_class=args.tr_num_per_class, val_num_per_class=args.val_num_per_class)
elif args.dataset in ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products']:
    split_idx_lst = dataset.get_idx_split()
else:
    if args.dataset in ['cora', 'citeseer', 'pubmed'] and args.protocol == 'semi':
        split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset, protocol=args.protocol)[0]
    elif args.dataset in ['chameleon', 'squirrel', 'film', 'cornell', 'texas', 'wisconsin'] and args.protocol == 'supervised':
        split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset, protocol=args.protocol)

if args.dataset == 'ogbn-proteins':
    edge_index_ = to_sparse_tensor(dataset.graph['edge_index'],
                                                   dataset.graph['edge_feat'], dataset.graph['num_nodes'])
    dataset.graph['node_feat'] = edge_index_.mean(dim=1)
    dataset.graph['edge_feat'] = None

n = dataset.graph['num_nodes']
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

# whether or not to symmetrize
if not args.directed and args.dataset != 'ogbn-proteins':
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

print(f"num nodes {n} | num classes {c} | num node feats {d}")

### Load model ###
model = parse_method(args, n, c, d, device)

if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins', 'yelp'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

if args.metric == 'rocauc':
    eval_func = eval_rocauc
elif args.metric == 'f1':
    eval_func = eval_f1
else:
    eval_func = eval_acc

logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)

train_times = []
train_mems = []

dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)
dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)

backup_data = copy.deepcopy(dataset)

### Training loop ###
for run in range(args.runs):
    if args.dataset in ['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel', 'film', 'cornell', 'texas', 'wisconsin'] and args.protocol == 'supervised':
        split_idx = split_idx_lst[run]
    else:
        split_idx = split_idx_lst
    train_idx = split_idx['train'].to(device)     
    
    valid_idx = torch.cat([train_idx, split_idx['valid'].to(device)], dim=-1)           
    dataset.graph['tr_edge_index'] = subgraph(train_idx, dataset.graph['edge_index'])[0]   # edge index for tr_node
    dataset.graph['va_edge_index'] = subgraph(valid_idx, dataset.graph['edge_index'])[0]   # edge index for tr_node and va_node
    assert dataset.graph['tr_edge_index'].shape[1] !=  dataset.graph['edge_index'].shape[1]

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
    best_val = float('-inf')
    
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        train_start = time.time()
        out = model(dataset.graph['node_feat'], dataset.graph['edge_index' if args.trans else 'tr_edge_index'], use_conv = args.conv_tr)

        if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins', 'yelp'):
            if dataset.label.shape[1] == 1:
                true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
            else:
                true_label = dataset.label
            loss = criterion(out[train_idx], true_label.squeeze(1)[train_idx].to(torch.float))
        
        else:
            out = F.log_softmax(out, dim=1)
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1).to(torch.float)
            loss = nn.KLDivLoss(reduction='batchmean')(out[train_idx], true_label[train_idx])

        loss.backward()
        optimizer.step()
        train_time = time.time() - train_start
        # train_mem = get_gpu_memory_map()[int(args.device)]
        # train_mems.append(train_mem)
        train_times.append(train_time)

        result = evaluate(model, dataset, split_idx, eval_func, criterion, args)
        logger.add_result(run, result[:-1])

        if result[1] > best_val:
            best_val = result[1]
            if args.dataset != 'ogbn-proteins':
                best_out = F.softmax(result[-1], dim=1)
            else:
                best_out = result[-1]

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%')
            if args.print_prop:
                pred = out.argmax(dim=-1, keepdim=True)
                print("Predicted proportions:", pred.unique(return_counts=True)[1].float() / pred.shape[0])
    logger.print_statistics(run)

results = logger.print_statistics()
train_time = sum(train_times) / len(train_times)
# train_mem = sum(train_mems) / len(train_mems)
para_num = count_parameters(model)

### Save results ###
filename = f'results/{args.dataset}.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    dataset_str = f'{args.dataset}'
    dataset_str += f'{args.sub_dataset},' if args.sub_dataset else ''
    write_obj.write(f"method{args.train_prop if args.rand_split else ''}:{args.method}(trans:{args.trans},cv_tr:{args.conv_tr},cv_va:{args.conv_va},cv_te:{args.conv_te}),\t lr:{args.lr},\t wd:{args.weight_decay},\t dpo:{args.dropout},\t l:{args.num_layers},\t o:{args.num_mps},\t hc:{args.hidden_channels},\t performance: {results.mean():.2f} $\pm$ {results.std():.2f},\t train time:{train_time: .6f},\t para_num:{para_num: .2f}\n")
                    