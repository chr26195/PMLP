from re import S
from xml.dom import xmlbuilder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import num_nodes, to_dense_adj
import numpy as np
import math
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from conv import *


# Alternative way to implement PMLP. There is only one line of difference with GCN.
class PMLP_GCN(nn.Module): 
    def __init__(self, in_channels, hidden_channels, out_channels, args, num_node):
        super(PMLP_GCN, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_node = num_node
        self.num_layers = args.num_layers
        self.ff_bias = True  

        self.bns = nn.BatchNorm1d(hidden_channels, affine=False, track_running_stats=False)
        self.activation = F.relu

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
        for _ in range(self.num_layers - 2): self.fcs.append(nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias)) #1s
        self.fcs.append(nn.Linear(hidden_channels, out_channels, bias=self.ff_bias)) #1
        self.reset_parameters()
    

    def reset_parameters(self):
        for mlp in self.fcs: 
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = x @ self.fcs[i].weight.t() 
            if not self.training:    # the only modified part
                x = gcn_conv(x, edge_index) 
            if self.ff_bias: x = x + self.fcs[i].bias
            if i != self.num_layers - 1:
                x = self.activation(self.bns(x))
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x