## Graph Neural Networks are Inherently Good Generalizers: Insights by Bridging GNNs and MLPs

This is the official code repository for "Graph Neural Networks are Inherently Good Generalizers: Insights by Bridging GNNs and MLPs"

Related materials: 
[paper](https://arxiv.org/pdf/2212.09034.pdf), [slides](https://github.com/chr26195/PMLP/blob/main/materials/slide_conference_version.pdf)

<img src="materials/illustration.png" width="900">

### What's news
[2023.04.11] We upload the (conference version) slide and add a "Quick Guide" section summarizing different ways to implement PMLP.

[2023.02.09] We release the early version of our codes for reproducibility (more detailed info will be updated soon).

## 0. What can I do with PMLP?
* Accelerate GNN training by modifying only a few lines of codes (see Quick Guide).
* Empower your MLP (or any other backbone models) by incorporating message passing / graph convolution in inference.
* Empower your GNN in some scenarios, e.g., datasets with many noisy structures, inductive learning setting.
* Simple and useful tool for research and scientific discovery.


## 1. Quick Guide
The implementation of PMLP is very simple, and can be plugged into your own pipeline by modifying only a few lines of codes. Here we introduce several different ways to implement PMLP.

### 1.1. Version A: Three Models (MLP/PMLP/GNN) in One Class
This is the default way to implement PMLP, which combines three models (MLP, PMLP, GNN) in one single class. The key part of this implementation is to add a `use_conv = True/False` parameter in the `self.forward()` function for any GNN classes. To implement PMLP, just set this parameter to be `False` in training and validation, and then reset it to be `True` in testing. For example:

``` python
# version A: three models (mlp, pmlp, gnn) in one class
class My_Own_GNN(nn.Module):
    ...
    def forward(self, x, edges, use_conv = True):
        ...
        x = self.feed_forward_layer(x) 
        if use_conv: 
            x = self.message_passing_layer(x, edges)  
        ...
        return x

my_own_gnn = My_Own_GNN()

# in the training loop
my_own_gnn.train()
for epoch in range(args.epochs):
    prediction = my_own_gnn(x, edge, use_conv = False)

# for inference
my_own_gnn.eval()
prediction = my_own_gnn(x, edge, use_conv = True)
```

This implementation is very flexible. If `use_conv = True` is both training and testing, the model is equivalent to the original GNN. And if `use_conv = False` is both training and testing, the model is equivalent to the original MLP (or other 'backbone' models). One can optionally let `use_conv = True` for validation by modifying the `evaluate()` function in `data_utils.py`, which could lead to better or worse performance depending on the specific task.

### 1.2. Version B: One Line of Code is All You Need
Here is the most simple way to implement PMLP, which leverages the (PyTorch) built-in parameter `self.training` to automatically specify when to use graph convolution.

``` python
# version B: one line of code is all you need
class My_Own_GNN(nn.Module):
    ...
    def forward(self, x, edges):
        ...
        x = self.feed_forward_layer(x) 
        if not self.training: # only modified part
            x = self.message_passing_layer(x, edges)  
        ...
        return x

my_own_gnn = My_Own_GNN()

# in the training loop
my_own_gnn.train()
for epoch in range(args.epochs):
    prediction = my_own_gnn(x, edges)

# for inference
my_own_gnn.eval()
prediction = my_own_gnn(x, edges)
```

### 1.3. Version C: Just Drop All Edges (But Leave Self-Loops Alone)
Another way to implement PMLP if to drop all edges but leave only self-loops in training such that message passing operation will not affect node representations. This is an extreme case of DropEdge [1]. Please refer to their [paper](https://arxiv.org/pdf/1907.10903.pdf) for more information. 

``` python
# version C: just drop all edges (but leave self-loops alone)
class My_Own_GNN(nn.Module):
    ...
    def forward(self, x, edges):
        ...
        x = self.feed_forward_layer(x) 
        x = self.message_passing_layer(x, edges)  
        ...
        return x

my_own_gnn = My_Own_GNN()

# create a graph with only self-loops
self_loops = torch.nonzero(torch.eye(node_numbers)).t()

# in the training loop
my_own_gnn.train()
for epoch in range(args.epochs):
    prediction = my_own_gnn(x, self_loops)

# for inference
my_own_gnn.eval()
prediction = my_own_gnn(x, edges)
```

### 1.4. Version D: Load Your Pretrained MLP
(coming soon)


[1] Rong, Yu, et al. "DropEdge: Towards Deep Graph Convolutional Networks on Node Classification." International Conference on Learning Representations 2020.


## 2. Run the Code
Step 1. Install the required package according to `requirements.txt`.

Step 2. Specify your own data path in `parse.py` and download the datasets.

Step 3. To run the training and evaluation, one can use the following scripts (for version A).

Step 4. Results will be saved in a folder named `results`

```shell
# GCN: use message passing in training, validation and testing
python main.py --dataset cora --method pmlp_gcn --protocol semi --lr 0.1 --weight_decay 0.01 --dropout 0.5 --num_layers 2 --hidden_channels 64 --induc --device 0 --conv_tr --conv_va --conv_te 

# PMLP_GCN: use message passing only in testing
python main.py --dataset cora --method pmlp_gcn --protocol semi --lr 0.1 --weight_decay 0.01 --dropout 0.5 --num_layers 2 --hidden_channels 64 --induc --device 0 --conv_te 

# MLP: not using message passing
python main.py --dataset cora --method pmlp_gcn --protocol semi --lr 0.1 --weight_decay 0.01 --dropout 0.5 --num_layers 2 --hidden_channels 64 --induc --device 0
```

`--induc` and `--trans` are used to specify inductive or transductive learning settings.

## Citation
If you are inspired by the paper or code, please consider citing our work
```bibtex
@inproceedings{yang2023pmlp,
title = {Graph Neural Networks are Inherently Good Generalizers: Insights by Bridging GNNs and MLPs},
author = {Chenxiao Yang and Qitian Wu and Jiahua Wang and Junchi Yan},
booktitle = {International Conference on Learning Representations (ICLR)},
year = {2023}
}
```