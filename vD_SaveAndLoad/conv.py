from re import S
from xml.dom import xmlbuilder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import num_nodes, to_dense_adj
import numpy as np
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
import math


def gcn_conv(h, edge_index):
    N = h.size(0)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)
    
    src, dst = edge_index
    deg = degree(dst, num_nodes=N)

    deg_src = deg[src].pow(-0.5) 
    deg_src.masked_fill_(deg_src == float('inf'), 0)
    deg_dst = deg[dst].pow(-0.5)
    deg_dst.masked_fill_(deg_dst == float('inf'), 0)
    edge_weight = deg_src * deg_dst

    a = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([N, N])).t()
    h_prime = a @ h 
    return h_prime


def conv_noloop(h, edge_index):
    N = h.size(0)
    edge_index, _ = remove_self_loops(edge_index)
    
    src, dst = edge_index
    deg = degree(dst, num_nodes=N)

    deg_src = deg[src].pow(-0.5) # 1/d^0.5(v_i)
    deg_src.masked_fill_(deg_src == float('inf'), 0)
    deg_dst = deg[dst].pow(-0.5) # 1/d^0.5(v_j)
    deg_dst.masked_fill_(deg_dst == float('inf'), 0)
    edge_weight = deg_src * deg_dst

    a = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([N, N])).t()
    h_prime = a @ h 
    return h_prime


def conv_rw(h, edge_index):
    N = h.size(0)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)
    
    src, dst = edge_index
    deg = degree(dst, num_nodes=N) 

    deg_src = deg[src].pow(-0) 
    deg_src.masked_fill_(deg_src == float('inf'), 0)
    deg_dst = deg[dst].pow(-1)
    deg_dst.masked_fill_(deg_dst == float('inf'), 0)
    edge_weight = deg_src * deg_dst

    a = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([N, N])).t()
    h_prime = a @ h 
    return h_prime


def conv_diff(h, edge_index):
    N = h.size(0)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)
    
    src, dst = edge_index
    deg = degree(dst, num_nodes=N) 

    deg_src = deg[src].pow(-0)
    deg_src.masked_fill_(deg_src == float('inf'), 0)
    deg_dst = deg[dst].pow(-1) 
    deg_dst.masked_fill_(deg_dst == float('inf'), 0)
    edge_weight = deg_src * deg_dst

    a = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([N, N])).t()
    h_prime = heat_kernel(a, h, 10) 
    return h_prime

def heat_kernel(a, h, k):
    h_prime = h, h_temp = h
    for i in range(1, k+1):
        h_temp = a @ h_temp / i
        h_prime += h_temp
    return h_prime / math.e


def conv_resi(h, edge_index, h_ori, alpha):
    N = h.size(0)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)
    
    src, dst = edge_index
    deg = degree(dst, num_nodes=N)

    deg_src = deg[src].pow(-0.5) # 1/d^0.5(v_i)
    deg_src.masked_fill_(deg_src == float('inf'), 0)
    deg_dst = deg[dst].pow(-0.5) # 1/d^0.5(v_j)
    deg_dst.masked_fill_(deg_dst == float('inf'), 0)
    edge_weight = deg_src * deg_dst

    a = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([N, N])).t()
    h_prime = (1. - alpha) * a @ h + alpha * h_ori
    return h_prime