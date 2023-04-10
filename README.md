## Graph Neural Networks are Inherently Good Generalizers: Insights by Bridging GNNs and MLPs

This is the official code repository for "Graph Neural Networks are Inherently Good Generalizers: Insights by Bridging GNNs and MLPs"

Related materials: 
[paper](https://arxiv.org/pdf/2212.09034.pdf), [slides](https://github.com/chr26195/PMLP/blob/main/materials/slide_conference_version.pdf)

<img src="materials/illustration.png" width="900">

### What's news
[2023.04.11] We upload the (conference version) slide and add a "Quick Guide" section summarizing key points in implementations.

[2023.02.09] We release the early version of our codes for reproducibility (more detailed info will be updated soon).

## 0. What can I do with PMLP?
* Accelerate GNN training by modifying only a few lines of codes (see Quick Guide).
* Empower your MLP (or any other backbone models) by incorporating message passing / graph convolution in inference.
* Empower your GNN in some scenarios, e.g., datasets with many noisy structures, inductive learning setting.
* Simple and useful tool for research and scientific discovery.


## 1. Quick Guide

### 1.1. Version A: Default PMLP Implementation
The implementation of PMLP is very simple, and can be plugged into your own pipeline by modifying only a few lines of codes. The key part of this implementation is to add a `use_conv = True/False` parameter in the `self.forward()` function for any GNN classes. This parameter is set to be `False` in training and validation, and reset to be `True` in testing. For example:

``` python
# version 1
class My_Own_GNN(nn.Module):
    ...
    def forward(self, x, edge_index, use_conv = True):
        ...
        x = self.feed_forward_layer(x) 
        if use_conv: 
            x = self.message_passing_layer(x, edge_index)  
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

This implementation is very flexible: If `use_conv` is `True` is both training and testing, the model is equivalent to the original GNN. And if `use_conv` is `False` is both training and testing, the model is equivalent to MLP (or other 'backbone' models).

### 1.2. Version B: One Line to Implement PMLP
Here is the most simple way to implement PMLP, which leverages the (PyTorch) built-in parameter `self.training` to automatically specify when to use graph convolution.

``` python
# version 2
class My_Own_GNN(nn.Module):
    ...
    def forward(self, x, edge_index):
        ...
        x = self.feed_forward_layer(x) 
        if not self.training: # only modified part
            x = self.message_passing_layer(x, edge_index)  
        ...
        return x

my_own_gnn.train()
for epoch in range(args.epochs):
    prediction = my_own_gnn(x, edge)

my_own_gnn.eval()
prediction = my_own_gnn(x, edge)
```

### 1.3. PMLP Extensions
(coming soon)

## 2. Run the Code
Step 1. Install the required package according to `requirements.txt`.

Step 2. Specify your own data path in `parse.py` and download the datasets.

Step 3. To run the training and evaluation, one can use the following scripts.

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