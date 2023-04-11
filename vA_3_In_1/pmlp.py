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


# Implementation of PMLP_GCN, which can become MLP or GCN depending on whether using message passing
class PMLP_GCN(nn.Module): 
    def __init__(self, in_channels, hidden_channels, out_channels, args, num_node):
        super(PMLP_GCN, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_node = num_node
        self.num_layers = args.num_layers
        self.ff_bias = True  # Use bias for FF layers in default

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

    def forward(self, x, edge_index, use_conv=True):
        for i in range(self.num_layers - 1):
            x = x @ self.fcs[i].weight.t() 
            if use_conv: x = gcn_conv(x, edge_index)  # Optionally replace 'gcn_conv' with other conv functions in conv.py
            if self.ff_bias: x = x + self.fcs[i].bias
            x = self.activation(self.bns(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = x @ self.fcs[-1].weight.t() 
        if use_conv: x = gcn_conv(x, edge_index)
        if self.ff_bias: x = x + self.fcs[-1].bias
        return x


# Implementation of PMLP_SGC, which can become MLP or SGC depending on whether using message passing
class PMLP_SGC(nn.Module): 
    def __init__(self, in_channels, hidden_channels, out_channels, args, num_node):
        super(PMLP_SGC, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_node = num_node
        self.num_layers = args.num_layers
        self.bns = nn.BatchNorm1d(hidden_channels, eps=1e-10, affine=False, track_running_stats=False)
        self.activation = F.relu
        self.ff_bias = True

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
        for _ in range(self.num_layers - 2): self.fcs.append(nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias)) #1s
        self.fcs.append(nn.Linear(hidden_channels, out_channels, bias=self.ff_bias)) #1
        self.reset_parameters()
    

    def reset_parameters(self):
        for mlp in self.fcs: 
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, x, edge_index, use_conv=True):
        for i in range(self.args.num_mps):
            if use_conv: x = gcn_conv(x, edge_index)

        for i in range(self.num_layers - 1):
            x = self.fcs[i](x) 
            x = self.activation(self.bns(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fcs[-1](x) 
        return x


# Implementation of PMLP_APP, which can become MLP or SGC depending on whether using message passing
class PMLP_APPNP(nn.Module): #residual connection
    def __init__(self, in_channels, hidden_channels, out_channels, args, num_node):
        super(PMLP_APPNP, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_node = num_node
        self.num_layers = args.num_layers
        self.ff_bias = True

        self.bns = nn.BatchNorm1d(hidden_channels, eps=1e-10, affine=False, track_running_stats=False)
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

    def forward(self, x, edge_index, use_conv=True):
        for i in range(self.num_layers - 1):
            x = self.fcs[i](x) 
            x = self.activation(self.bns(x))
            x = F.dropout(x, p=self.dropout, training=self.training)    
        x = self.fcs[-1](x) 
        for i in range(self.args.num_mps):
            if use_conv: x = gcn_conv(x, edge_index)    
        return x
    


# The rest models are used for additional experiments in the paper

# Implementation of PMLP_GCNII, which can become ResNet (MLP with residual connections) or GCNII depending on whether using message passing
class PMLP_GCNII(nn.Module): #GCNII
    def __init__(self, in_channels, hidden_channels, out_channels, args, num_node):
        super(PMLP_GCNII, self).__init__()
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

    def forward(self, x, edge_index, use_conv=True):
        x = x @ self.fcs[0].weight.t()
        x = self.activation(self.bns(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x_ = x.clone()

        for i in range(1, self.num_layers - 1):
            x = x * (1. - 0.5 / i) + x @ self.fcs[i].weight.t() * (0.5 / i) 
            if use_conv: x = conv_resi(x, edge_index, x_)
            else: x = 0.9 * x + 0.1 * x_
            x = self.activation(self.bns(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x =  x @ self.fcs[-1].weight.t() 
        if use_conv: x = gcn_conv(x, edge_index)
        return x


class PMLP_JKNet(nn.Module): #JKNET(concatation pooling)
    def __init__(self, in_channels, hidden_channels, out_channels, args, num_node):
        super(PMLP_JKNet, self).__init__()
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
        self.fcs.append(nn.Linear(hidden_channels * (self.num_layers - 1), out_channels, bias=self.ff_bias)) #1
        self.reset_parameters()

    def reset_parameters(self):
        for mlp in self.fcs: 
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, x, edge_index, use_conv=True):
        xs = []
        for i in range(0, self.num_layers - 1):
            x = x @ self.fcs[i].weight.t() 
            if use_conv: x = gcn_conv(x, edge_index)
            x = self.activation(self.bns(x))
            xs.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = torch.cat(xs, dim=-1)
        x =  x @ self.fcs[-1].weight.t() 
        return x

class PMLP_SGCres(nn.Module): #SGC with residual connections
    def __init__(self, in_channels, hidden_channels, out_channels, args, num_node):
        super(PMLP_SGCres, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_node = num_node
        self.num_layers = args.num_layers
        self.bns = nn.BatchNorm1d(hidden_channels, eps=1e-10, affine=False, track_running_stats=False)
        self.activation = F.relu
        self.ff_bias = True

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
        for _ in range(self.num_layers - 2): self.fcs.append(nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias)) #1s
        self.fcs.append(nn.Linear(hidden_channels, out_channels, bias=self.ff_bias)) #1
        self.reset_parameters()
    

    def reset_parameters(self):
        for mlp in self.fcs: 
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, x, edge_index, use_conv=True):
        x_ = x.clone()
        for i in range(self.args.num_mps):
            if use_conv: x = conv_resi(x, edge_index, x_)

        for i in range(self.num_layers - 1):
            x = self.fcs[i](x) 
            x = self.activation(self.bns(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fcs[-1](x) 
        return x



class PMLP_SGCresinf(nn.Module): #SGC with residual connections (in test but not in train)
    def __init__(self, in_channels, hidden_channels, out_channels, args, num_node):
        super(PMLP_SGCresinf, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_node = num_node
        self.num_layers = args.num_layers
        self.bns = nn.BatchNorm1d(hidden_channels, eps=1e-10, affine=False, track_running_stats=False)
        self.activation = F.relu
        self.ff_bias = True

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
        for _ in range(self.num_layers - 2): self.fcs.append(nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias)) #1s
        self.fcs.append(nn.Linear(hidden_channels, out_channels, bias=self.ff_bias)) #1
        self.reset_parameters()
    

    def reset_parameters(self):
        for mlp in self.fcs: 
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, x, edge_index, use_conv=True):
        x_ = x.clone()
        for i in range(self.args.num_mps):
            if use_conv: x = conv_resi(x, edge_index, x_)
            else: x =  gcn_conv(x, edge_index)

        for i in range(self.num_layers - 1):
            x = self.fcs[i](x) 
            x = self.activation(self.bns(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fcs[-1](x) 
        return x
    
    
class PMLP_APPNPres(nn.Module): 
    def __init__(self, in_channels, hidden_channels, out_channels, args, num_node):
        super(PMLP_APPNPres, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_node = num_node
        self.num_layers = args.num_layers
        self.ff_bias = True

        self.bns = nn.BatchNorm1d(hidden_channels, eps=1e-10, affine=False, track_running_stats=False)
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

    def forward(self, x, edge_index, use_conv=True):
        for i in range(self.num_layers - 1):
            x = self.fcs[i](x) 
            x = self.activation(self.bns(x))
            x = F.dropout(x, p=self.dropout, training=self.training)    
        x = self.fcs[-1](x) 
        x_ = x.clone()
        for i in range(self.args.num_mps):
            if use_conv: x = conv_resi(x, edge_index, x_)
        return x


class PMLP_APPNPresinf(nn.Module): 
    def __init__(self, in_channels, hidden_channels, out_channels, args, num_node):
        super(PMLP_APPNPresinf, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_node = num_node
        self.num_layers = args.num_layers
        self.ff_bias = True

        self.bns = nn.BatchNorm1d(hidden_channels, eps=1e-10, affine=False, track_running_stats=False)
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

    def forward(self, x, edge_index, use_conv=True):
        for i in range(self.num_layers - 1):
            x = self.fcs[i](x) 
            x = self.activation(self.bns(x))
            x = F.dropout(x, p=self.dropout, training=self.training)    
        x = self.fcs[-1](x) 
        x_ = x.clone()
        for i in range(self.args.num_mps):
            if use_conv: x = conv_resi(x, edge_index, x_)
            else: x =  gcn_conv(x, edge_index)
        return x