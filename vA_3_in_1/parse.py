from pmlp import *
from data_utils import normalize

def parse_method(args, n, c, d, device):
    if args.method == 'pmlp_gcn': model = PMLP_GCN(d, args.hidden_channels, c, args, n).to(device)
    elif args.method == 'pmlp_gcnii': model = PMLP_GCNII(d, args.hidden_channels, c, args, n).to(device)
    elif args.method == 'pmlp_jknet': model = PMLP_JKNet(d, args.hidden_channels, c, args, n).to(device)
    
    elif args.method == 'pmlp_sgc': model = PMLP_SGC(d, args.hidden_channels, c, args, n).to(device)
    elif args.method == 'pmlp_sgc_res': model = PMLP_SGCres(d, args.hidden_channels, c, args, n).to(device)
    elif args.method == 'pmlp_sgc_resinf': model = PMLP_SGCresinf(d, args.hidden_channels, c, args, n).to(device)
    
    elif args.method == 'pmlp_appnp': model = PMLP_APPNP(d, args.hidden_channels, c, args, n).to(device)
    elif args.method == 'pmlp_appnp_res': model = PMLP_APPNPres(d, args.hidden_channels, c, args, n).to(device)
    elif args.method == 'pmlp_appnp_resinf': model = PMLP_APPNPresinf(d, args.hidden_channels, c, args, n).to(device)

    else: raise NotImplementedError
    return model


def parser_add_main_args(parser):
    # dataset and evaluation
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--runs', type=int, default=5,
                        help='number of distinct runs')
    parser.add_argument('--directed', action='store_true',
                        help='set to not symmetrize adjacency')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    parser.add_argument('--protocol', type=str, default='semi',
                        help='protocol for cora datasets, semi or supervised')
    parser.add_argument('--rand_split', action='store_true', help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--tr_num_per_class', type=int, default=20, help='training nodes randomly selected')
    parser.add_argument('--val_num_per_class', type=int, default=30, help='valid nodes randomly selected')
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc', 'f1'],
                        help='evaluation metric')

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--cached', action='store_true',
                        help='set to use faster sgc')
    parser.add_argument('--print_prop', action='store_true',
                        help='print proportions of predicted class')

    # -------------------------------------------------------------------------------------            
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2, help='number of feed-forward layers')
    parser.add_argument('--num_mps', type=int, default=2, help='number of message passing layers') # ignore this hyperparameter for GCN-style GNNs

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--method', default='pmlp_gcn') # options: [pmlp_gcn, pmlp_sgc, pmlp_appnp, pmlp_gcnii, pmlp_jknet, ....]
    parser.add_argument('--conv_tr', action='store_true')
    parser.add_argument('--conv_va', action='store_true')
    parser.add_argument('--conv_te', action='store_true')
    parser.add_argument('--trans', action='store_true') # transductive (whole graph structure for train)
    parser.add_argument('--induc', action='store_true') # inductive (graph structure of training node for train) setting
    # -------------------------------------------------------------------------------------            