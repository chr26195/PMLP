## Graph Neural Networks are Inherently Good Generalizers: Insights by Bridging GNNs and MLPs

This is the official code repository for "Graph Neural Networks are Inherently Good Generalizers: Insights by Bridging GNNs and MLPs"

Related materials: 
[paper](https://arxiv.org/pdf/2212.09034.pdf)

### What's news

[2023.02.09] We release the early version of our codes for reproducibility (more detailed info will be updated soon).

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
      @inproceedings{yang2023pmlp,
      title = {Graph Neural Networks are Inherently Good Generalizers: Insights by Bridging GNNs and MLPs},
      author = {Chenxiao Yang and Qitian Wu and Jiahua Wang and Junchi Yan},
      booktitle = {International Conference on Learning Representations (ICLR)},
      year = {2023}
      }
```