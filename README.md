## Graph Neural Networks are Inherently Good Generalizers: Insights by Bridging GNNs and MLPs

This is the official code repository for "Graph Neural Networks are Inherently Good Generalizers: Insights by Bridging GNNs and MLPs". It is highly recommended to read the "Quick Guide" section before perusing the codes.

Related materials: 
[paper](https://arxiv.org/pdf/2212.09034.pdf), [slides](https://github.com/chr26195/PMLP/blob/main/materials/slide_conference_version.pdf), [poster](https://github.com/chr26195/PMLP/blob/main/materials/poster.pdf)

<img src="materials/illustration.png" width="900">

### What's news

[2023.04.11] We upload the (conference version) slide and add a "Quick Guide" section summarizing different ways to implement PMLP.

[2023.02.09] We release the early version of our codes for reproducibility (more detailed info will be updated soon).

## 0. What Could We Do with PMLP?
* Accelerate GNN training by modifying only a few lines of codes.
* Empower MLP (or any other backbone models) by incorporating message passing / graph convolution in inference.
* Empower GNN in some scenarios, e.g., datasets with many noisy structures.
* Simple and useful tool for research and scientific discovery.


## 1. Quick Guide
The implementation of PMLP is very simple, and can be plugged into one's own pipeline by modifying only a few lines of codes. **The key idea of PMLP is to just remove message passing modules in GNNs during training.** We allow the model after removal of message passing layers to be other models, such as ResNet (corresponding to GCNII) and MLP+JK (corresponding to JKNet). Here we introduce several different ways to implement PMLP and discuss their advantages and limitations.

### 1.1. Version A: Three Models (MLP / PMLP / GNN) in One Class
This is the default way and go-to choice to implement PMLP, which combines three models (MLP, PMLP, GNN) in one single class. The key idea of this implementation is to add a `use_conv = True/False` parameter in the `self.forward()` function for any GNN classes. To implement PMLP, just set this parameter to be `False` in training and validation, and then reset it to be `True` in testing. For example:

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

This implementation is very flexible. If `use_conv = True` is both training and testing, the model is equivalent to the original GNN. And if `use_conv = False` is both training and testing, the model is equivalent to the original MLP (or other 'backbone' models). One can optionally let `use_conv = True` for validation by modifying the `evaluate()` function in `data_utils.py`, which could lead to better or worse performance depending on the specific task and could be treated as a hyperparameter to tune. In default, we do not use message passing in validation such that the model selection process of PMLP is also equivalent to MLP. Many extensions of PMLP can be developed upon this implementation for adapting to new tasks or further boosting performance. For example, we can specify `def if_use_conv(*args)` as a function of different inputs (e.g., whether the model is training, whether the loss function has reached a certain threshold, current training epoch, etc.) to control concretely when and where to use message passing layers.

### 1.2. Version B: One Line of Code is All You Need
Here is the most simple way to implement PMLP, which only requires modifying only one line of code in your own GNN pipeline. This implementation leverages the (PyTorch) built-in parameter `self.training` to automatically specify when to use graph convolution.

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

One limitation of this implementation is that it enforces using message passing (i.e., the GNN architecture) for validation (since usually we use `model.eval()` before validation and `self.training` is automatically set to be `False`). However, this is totally acceptable in most scenarios since validation is not the bottleneck of training efficiency and using message passing in validation often improves performance. 

### 1.3. Version C: Just Drop All Edges (But Leave Self-Loops Alone)
One equivalent way to implement PMLP is to drop all edges but leave only self-loops in training such that message passing operation will not affect node representations. This is an extreme case of DropEdge. Please refer to their [paper](https://arxiv.org/pdf/1907.10903.pdf) for more information. 

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

This implementation makes PMLP very easy to adapt since one does not need to modify anything on the original GNN class. ***But please note that this implementation is LESS efficient than other PMLP versions as we have observed in practice***, presumably because it still relies on the messaga passing layer. Please also be careful that it is possible that for some specific message passing implementations, the corresponding transition matrix for self-loops is not an indentity matrix, and then this version would not be exactly equivalent to PMLP.

### 1.4. Version D: Load Pretrained MLP
Another equivalent way to implement PMLP is to define two models, i.e., MLP model and GNN model. We first pretrain the MLP model, save the `state_dict()`, then load it to the GNN model, and finally use it directly for inference or do whatever we want on top of it. 

``` python
# version D: Load Pretrained MLP
class My_Own_GNN(nn.Module):
    ...
    def forward(self, x, edges):
        ...
        x = self.feed_forward_layer(x) 
        x = self.message_passing_layer(x, edges)  
        ...
        return x

class Corresponding_MLP(nn.Module):
    ...
    def forward(self, x, edges):
        ...
        x = self.feed_forward_layer(x) 
        ...
        return x

model_mlp = Corresponding_MLP()
model_gnn = My_Own_GNN()

# in the training loop
model_mlp.train()
for epoch in range(args.epochs):
    prediction = model_mlp(x)
    ...
torch.save(model_mlp.state_dict(), PATH)

# for inference
model_gnn.load_state_dict(PATH)
model_gnn.eval()
prediction = model_gnn(x, edges)
```

This version is more complicated than others in terms of implementation but could be suitable for empowering pre-trained MLP (or any other backbone models) by incorporating message passing / graph convolution in inference and some other scenarios.

## 2. Bag of Tricks to Boost PMLP
(coming soon)

## 3. Run the Code
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

## Citation and Contact
If you found our codes useful, please consider giving this project a star. If you are inspired by PMLP in your research, please consider citing our work
```bibtex
@inproceedings{yang2023pmlp,
    title = {Graph Neural Networks are Inherently Good Generalizers: Insights by Bridging GNNs and MLPs},
    author = {Chenxiao Yang and Qitian Wu and Jiahua Wang and Junchi Yan},
    booktitle = {International Conference on Learning Representations (ICLR)},
    year = {2023}
}
```

Wanna further accelerate GNN in inference or gain insights on how GNNs capture data geometry? Please check our previous work "Geometric Knowledge Distillation: Topology Compression for Graph Neural Networks" in NeurIPS 2022. ([paper](https://arxiv.org/pdf/2210.13014.pdf), [code](https://github.com/chr26195/GKD)).
```bibtex
@inproceedings{yang2022geometric,
      title = {Geometric Knowledge Distillation: Topology Compression for Graph Neural Networks},
      author = {Chenxiao Yang and Qitian Wu and Junchi Yan},
      booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
      year = {2022}
}
```


