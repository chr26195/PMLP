## Graph Neural Networks are Inherently Good Generalizers: Insights by Bridging GNNs and MLPs

This is the official code repository for "Graph Neural Networks are Inherently Good Generalizers: Insights by Bridging GNNs and MLPs"

**Abstract:** Graph neural networks (GNNs), as the de-facto model class for representation learning on graphs, are built upon the multi-layer perceptrons (MLP) architecture with additional message passing layers to allow features to flow across nodes. While conventional wisdom largely attributes the success of GNNs to their advanced expressivity for learning desired functions on nodes' ego-graphs, we conjecture that this is not the main cause of GNNs' superiority in node prediction tasks. This paper pinpoints the major source of GNNs' performance gain to their intrinsic generalization capabilities, by introducing an intermediate model class dubbed as P(ropagational)MLP, which is identical to standard MLP in training, and then adopt GNN's architecture in testing. Intriguingly, we observe that PMLPs consistently perform on par with (or even exceed) their GNN counterparts across ten benchmarks and different experimental settings, despite the fact that PMLPs share the same (trained) weights with poorly-performed MLP. This critical finding opens a door to a brand new perspective for understanding the power of GNNs, and allow bridging GNNs and MLPs for dissecting their generalization behaviors. As an initial step to analyze PMLP, we show its essential difference with MLP at infinite-width limit lies in the NTK feature map in the post-training stage. Moreover, though MLP and PMLP cannot extrapolate non-linear functions for extreme OOD data, PMLP has more freedom to generalize near the training support. 

Related materials: 
[paper](https://arxiv.org/pdf/2212.09034.pdf)

### Use the Code
1. Install the required package according to `requirements.txt`.
2. Specify your own data path in `parse.py` and download the datasets.
3. To run the training and evaluation on eight datasets we used, one can use the following scripts.
4. Results will be saved in a folder named `results`

```shell
# GCN: use message passing in training, validation and testing
python main.py --dataset cora --method pmlp_gcn --protocol semi --lr 0.1 --weight_decay 0.01 --dropout 0.5 --num_layers 2 --hidden_channels 64 --induc --device 0 --conv_tr --conv_va --conv_te 

# PMLP_GCN: use message passing only in testing
python main.py --dataset cora --method pmlp_gcn --protocol semi --lr 0.1 --weight_decay 0.01 --dropout 0.5 --num_layers 2 --hidden_channels 64 --induc --device 0 --conv_te 

# MLP: not using message passing
python main.py --dataset cora --method pmlp_gcn --protocol semi --lr 0.1 --weight_decay 0.01 --dropout 0.5 --num_layers 2 --hidden_channels 64 --induc --device 0
```

### Citation
If you find our codes useful, please consider citing our work
```bibtex
      @inproceedings{yang2023geometric,
      title = {Geometric Knowledge Distillation: Topology Compression for Graph Neural Networks},
      author = {Chenxiao Yang and Qitian Wu and Junchi Yan},
      booktitle = {Inter (NeurIPS)},
      year = {2023}
      }
```

### ACK
The pipeline for training and preprocessing is developed on basis of the Non-Homophilous Benchmark project.