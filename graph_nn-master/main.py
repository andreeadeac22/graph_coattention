import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import time
import math
import torch
import torch.utils
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

from utils import split_ids
from graph_coattn import CoAttentionMessagePassingNetwork

print('using torch', torch.__version__)

# Experiment parameters
parser = argparse.ArgumentParser(description='Graph Convolutional Networks')
parser.add_argument('-D', '--dataset', type=str, default='PROTEINS')
parser.add_argument('-M', '--model', type=str, default='gcn', choices=['gcn', 'unet', 'mgcn', 'coattn'])
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--lr_decay_steps', type=str, default='25,35', help='learning rate')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
parser.add_argument('-d', '--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('-f', '--filters', type=str, default='64,64,64', help='number of filters in each layer')
parser.add_argument('-K', '--filter_scale', type=int, default=1, help='filter scale (receptive field size), must be > 0; 1 for GCN, >1 for ChebNet')
parser.add_argument('--n_hidden', type=int, default=0,
                    help='number of hidden units in a fully connected layer after the last conv layer')
parser.add_argument('--n_hidden_edge', type=int, default=32,
                    help='number of hidden units in a fully connected layer of the edge prediction network')
parser.add_argument('--degree', action='store_true', default=False, help='use one-hot node degree features')
parser.add_argument('--epochs', type=int, default=40, help='number of epochs')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--bn', action='store_true', default=False, help='use BatchNorm layer')
parser.add_argument('--folds', type=int, default=10, help='number of cross-validation folds (1 for COLORS and TRIANGLES and 10 for other datasets)')
parser.add_argument('-t', '--threads', type=int, default=0, help='number of threads to load data')
parser.add_argument('--log_interval', type=int, default=10, help='interval (number of batches) of logging')
#parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
parser.add_argument('--seed', type=int, default=111, help='random seed')
parser.add_argument('--shuffle_nodes', action='store_true', default=False, help='shuffle nodes for debugging')
parser.add_argument('-g', '--torch_geom', action='store_true', default=False, help='use PyTorch Geometric')
parser.add_argument('-a', '--adj_sq', action='store_true', default=False,
                    help='use A^2 instead of A as an adjacency matrix')
parser.add_argument('-s', '--scale_identity', action='store_true', default=False,
                    help='use 2I instead of I for self connections')
parser.add_argument('-v', '--visualize', action='store_true', default=False,
                    help='only for unet: save some adjacency matrices and other data as images')
parser.add_argument('-c', '--use_cont_node_attr', action='store_true', default=False,
                    help='use continuous node attributes in addition to discrete ones')

args = parser.parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.torch_geom:
    from torch_geometric.datasets import TUDataset
    import torch_geometric.transforms as T

args.filters = list(map(int, args.filters.split(',')))
args.lr_decay_steps = list(map(int, args.lr_decay_steps.split(',')))

for arg in vars(args):
    print(arg, getattr(args, arg))

n_folds = args.folds  # train,val,test splits for COLORS and TRIANGLES and 10-fold cross validation for other datasets
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
rnd_state = np.random.RandomState(args.seed)


if not args.torch_geom:
    from CustomData import GraphData, DataReader

from graphconv_layers import *


def collate_batch(batch):
    '''
    Creates a batch of same size graphs by zero-padding node features and adjacency matrices up to
    the maximum number of nodes in the CURRENT batch rather than in the entire dataset.
    Graphs in the batches are usually much smaller than the largest graph in the dataset, so this method is fast.
    :param batch: batch in the PyTorch Geometric format or [node_features*batch_size, A*batch_size, label*batch_size]
    :return: [node_features, A, graph_support, N_nodes, label]
    '''
    B = len(batch)
    if args.torch_geom:
        N_nodes = [len(batch[b].x) for b in range(B)]
        C = batch[0].x.shape[1]
    else:
        N_nodes = [len(batch[b][1]) for b in range(B)]
        C = batch[0][0].shape[1]
    N_nodes_max = int(np.max(N_nodes))

    graph_support = torch.zeros(B, N_nodes_max)
    A = torch.zeros(B, N_nodes_max, N_nodes_max)
    x = torch.zeros(B, N_nodes_max, C)
    for b in range(B):
        if args.torch_geom:
            x[b, :N_nodes[b]] = batch[b].x
            A[b].index_put_((batch[b].edge_index[0], batch[b].edge_index[1]), torch.Tensor([1]))
        else:
            x[b, :N_nodes[b]] = batch[b][0]
            A[b, :N_nodes[b], :N_nodes[b]] = batch[b][1]
        graph_support[b][:N_nodes[b]] = 1  # mask with values of 0 for dummy (zero padded) nodes, otherwise 1

    N_nodes = torch.from_numpy(np.array(N_nodes)).long()
    labels = torch.from_numpy(np.array([batch[b].y if args.torch_geom else batch[b][2] for b in range(B)])).long()
    return [x, A, graph_support, N_nodes, labels]


is_regression = args.dataset in ['COLORS-3', 'TRIANGLES']  # other datasets can be for the regression task (see their README.txt)
transforms = []  # for PyTorch Geometric
if args.dataset in ['COLORS-3', 'TRIANGLES']:
    assert n_folds == 1, 'use train, val and test splits for these datasets'
    assert args.use_cont_node_attr, 'node attributes should be used for these datasets'

    if args.torch_geom:
        # Class to read note attention from DS_node_attributes.txt
        class HandleNodeAttention(object):
            def __call__(self, data):
                if args.dataset == 'COLORS-3':
                    data.attn = torch.softmax(data.x[:, 0], dim=0)
                    data.x = data.x[:, 1:]
                else:
                    data.attn = torch.softmax(data.x, dim=0)
                    data.x = None
                return data

        transforms.append(HandleNodeAttention())
else:
    assert n_folds == 10, '10-fold cross-validation should be used for other datasets'

print('Regression={}'.format(is_regression))
print('Loading data')

if is_regression:
    def loss_fn(output, target, reduction='mean'):
        loss = (target.float().squeeze() - output.squeeze()) ** 2
        return loss.sum() if reduction == 'sum' else loss.mean()

    predict_fn = lambda output: output.round().long().detach().cpu()
else:
    loss_fn = F.cross_entropy
    predict_fn = lambda output: output.max(1, keepdim=True)[1].detach().cpu()

if args.torch_geom:
    if args.degree:
        if args.dataset == 'TRIANGLES':
            max_degree = 14
        else:
            raise NotImplementedError('max_degree value should be specified in advance. '
                                      'Try running without --torch_geom (-g) and look at dataset statistics printed out by our code.')

    if args.degree:
        transforms.append(T.OneHotDegree(max_degree=max_degree, cat=False))

    dataset = TUDataset('./data/%s/' % args.dataset, name=args.dataset,
                        use_node_attr=args.use_cont_node_attr,
                        transform=T.Compose(transforms))
    train_ids, test_ids = split_ids(args, rnd_state.permutation(len(dataset)), folds=n_folds)

else:
    datareader = DataReader(args=args, data_dir='./data/%s/' % args.dataset,
                            rnd_state=rnd_state,
                            folds=n_folds,
                            use_cont_node_attr=args.use_cont_node_attr)

acc_folds = []

for fold_id in range(n_folds):

    loaders = []
    for split in ['train', 'test']:
        if args.torch_geom:
            gdata = dataset[torch.from_numpy((train_ids if split.find('train') >= 0 else test_ids)[fold_id])]
        else:
            gdata = GraphData(fold_id=fold_id,
                              datareader=datareader,
                              split=split)

        loader = DataLoader(gdata,
                            batch_size=args.batch_size,
                            shuffle=split.find('train') >= 0,
                            num_workers=args.threads,
                            collate_fn=collate_batch)
        loaders.append(loader)

    print('\nFOLD {}/{}, train {}, test {}'.format(fold_id + 1, n_folds, len(loaders[0].dataset), len(loaders[1].dataset)))

    if args.model == 'gcn':
        model = GCN(in_features=loaders[0].dataset.num_features,
                    out_features=1 if is_regression else loaders[0].dataset.num_classes,
                    n_hidden=args.n_hidden,
                    filters=args.filters,
                    K=args.filter_scale,
                    bnorm=args.bn,
                    dropout=args.dropout,
                    adj_sq=args.adj_sq,
                    scale_identity=args.scale_identity).to(args.device)
    elif args.model == 'unet':
        model = GraphUnet(in_features=loaders[0].dataset.num_features,
                          out_features=1 if is_regression else loaders[0].dataset.num_classes,
                          n_hidden=args.n_hidden,
                          filters=args.filters,
                          K=args.filter_scale,
                          bnorm=args.bn,
                          dropout=args.dropout,
                          adj_sq=args.adj_sq,
                          scale_identity=args.scale_identity,
                          shuffle_nodes=args.shuffle_nodes,
                          visualize=args.visualize).to(args.device)
    elif args.model == 'mgcn':
        model = MGCN(in_features=loaders[0].dataset.num_features,
                     out_features=1 if is_regression else loaders[0].dataset.num_classes,
                     n_relations=2,
                     n_hidden=args.n_hidden,
                     n_hidden_edge=args.n_hidden_edge,
                     filters=args.filters,
                     K=args.filter_scale,
                     bnorm=args.bn,
                     dropout=args.dropout,
                     adj_sq=args.adj_sq,
                     scale_identity=args.scale_identity).to(args.device)
    elif args.model == 'coattn':
        model = CoAttentionMessagePassingNetwork(
            in_features=loaders[0].dataset.num_features,
            out_features=1 if is_regression else loaders[0].dataset.num_classes,
            shuffle_nodes=args.shuffle_nodes,
            visualize=args.visualize).to(args.device)
    else:
        raise NotImplementedError(args.model)

    print('\nInitialize model')
    print(model)
    train_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    print('N trainable parameters:', np.sum([p.numel() for p in train_params]))

    optimizer = optim.Adam(train_params, lr=args.lr, weight_decay=args.wd, betas=(0.5, 0.999))
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_decay_steps, gamma=0.1)

    # Normalization of continuous node features
    # if args.use_cont_node_attr:
    #     x = []
    #     for batch_idx, data in enumerate(loaders[0]):
    #         if args.torch_geom:
    #             node_attr_dim = loaders[0].dataset.props['node_attr_dim']
    #         x.append(data[0][:, :, :node_attr_dim].view(-1, node_attr_dim).data)
    #     x = torch.cat(x)
    #     mn, sd = torch.mean(x, dim=0).to(args.device), torch.std(x, dim=0).to(args.device) + 1e-5
    #     print(mn, sd)
    # else:
    #     mn, sd = 0, 1

    # def norm_features(x):
    #     x[:, :, :node_attr_dim] = (x[:, :, :node_attr_dim] - mn) / sd

    def train(train_loader):
        scheduler.step()
        model.train()
        start = time.time()
        train_loss, n_samples = 0, 0
        for batch_idx, data in enumerate(train_loader):
            for i in range(len(data)):
                data[i] = data[i].to(args.device)
            # if args.use_cont_node_attr:
            #     data[0] = norm_features(data[0])
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, data[4])
            loss.backward()
            optimizer.step()
            time_iter = time.time() - start
            train_loss += loss.item() * len(output)
            n_samples += len(output)
            if batch_idx % args.log_interval == 0 or batch_idx == len(train_loader) - 1:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (avg: {:.6f}) \tsec/iter: {:.4f}'.format(
                    epoch + 1, n_samples, len(train_loader.dataset),
                    100. * (batch_idx + 1) / len(train_loader), loss.item(), train_loss / n_samples,
                    time_iter / (batch_idx + 1)))


    def test(test_loader):
        model.eval()
        start = time.time()
        test_loss, correct, n_samples = 0, 0, 0
        for batch_idx, data in enumerate(test_loader):
            for i in range(len(data)):
                data[i] = data[i].to(args.device)
            # if args.use_cont_node_attr:
            #     data[0] = norm_features(data[0])
            output = model(data)
            loss = loss_fn(output, data[4], reduction='sum')
            test_loss += loss.item()
            n_samples += len(output)
            pred = predict_fn(output)

            correct += pred.eq(data[4].detach().cpu().view_as(pred)).sum().item()

        acc = 100. * correct / n_samples
        print('Test set (epoch {}): Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) \tsec/iter: {:.4f}\n'.format(
            epoch + 1,
            test_loss / n_samples,
            correct,
            n_samples,
            acc, (time.time() - start) / len(test_loader)))
        return acc

    for epoch in range(args.epochs):
        train(loaders[0])  # no need to evaluate after each epoch
    acc = test(loaders[1])
    acc_folds.append(acc)

print(acc_folds)
print('{}-fold cross validation avg acc (+- std): {} ({})'.format(n_folds, np.mean(acc_folds), np.std(acc_folds)))
