import torch
import torch.nn as nn
import numpy as np

cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NN layers and models
class GraphConv(nn.Module):
    '''
    Graph Convolution Layer according to (T. Kipf and M. Welling, ICLR 2017) if K<=1
    Chebyshev Graph Convolution Layer according to (M. Defferrard, X. Bresson, and P. Vandergheynst, NIPS 2017) if K>1
    Additional tricks (power of adjacency matrix and weighted self connections) as in the Graph U-Net paper
    '''

    def __init__(self,
                 in_features,
                 out_features,
                 n_relations=1,  # number of relation types (adjacency matrices)
                 K=1,  # GCN is K<=1, else ChebNet
                 activation=None,
                 bnorm=False,
                 adj_sq=False,
                 scale_identity=False):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features=in_features * K * n_relations, out_features=out_features)
        self.n_relations = n_relations
        assert K > 0, ('filter scale must be greater than 0', K)
        self.K = K
        self.activation = activation
        self.bnorm = bnorm
        if self.bnorm:
            self.bn = nn.BatchNorm1d(out_features)
        self.adj_sq = adj_sq
        self.scale_identity = scale_identity

    def chebyshev_basis(self, L, X, K):
        if K > 1:
            Xt = [X]
            Xt.append(torch.bmm(L, X))  # B,N,F
            for k in range(2, K):
                Xt.append(2 * torch.bmm(L, Xt[k - 1]) - Xt[k - 2])  # B,N,F
            Xt = torch.cat(Xt, dim=2)  # B,N,K,F
            return Xt
        else:
            # GCN
            assert K == 1, K
            return torch.bmm(L, X)  # B,N,1,F

    def laplacian_batch(self, A):
        batch, N = A.shape[:2]
        if self.adj_sq:
            A = torch.bmm(A, A)  # use A^2 to increase graph connectivity
        A_hat = A
        if self.K < 2 or self.scale_identity:
            I = torch.eye(N).unsqueeze(0).to(cuda_device)
            if self.scale_identity:
                I = 2 * I  # increase weight of self connections
            if self.K < 2:
                A_hat = A + I
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        return L

    def forward(self, data):
        x, A, mask = data[:3]
        # print('in', x.shape, torch.sum(torch.abs(torch.sum(x, 2)) > 0))
        if len(A.shape) == 3:
            A = A.unsqueeze(3)
        x_hat = []

        for rel in range(self.n_relations):
            L = self.laplacian_batch(A[:, :, :, rel])
            x_hat.append(self.chebyshev_basis(L, x, self.K))
        x = self.fc(torch.cat(x_hat, 2))

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(2)

        x = x * mask  # to make values of dummy nodes zeros again, otherwise the bias is added after applying self.fc which affects node embeddings in the following layers

        if self.bnorm:
            x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.activation is not None:
            x = self.activation(x)
        return (x, A, mask)


class GCN(nn.Module):
    '''
    Baseline Graph Convolutional Network with a stack of Graph Convolution Layers and global pooling over nodes.
    '''

    def __init__(self,
                 in_features,
                 out_features,
                 filters=[64, 64, 64],
                 K=1,
                 bnorm=False,
                 n_hidden=0,
                 dropout=0.2,
                 adj_sq=False,
                 scale_identity=False):
        super(GCN, self).__init__()

        # Graph convolution layers
        self.gconv = nn.Sequential(*([GraphConv(in_features=in_features if layer == 0 else filters[layer - 1],
                                                out_features=f,
                                                K=K,
                                                activation=nn.ReLU(inplace=True),
                                                bnorm=bnorm,
                                                adj_sq=adj_sq,
                                                scale_identity=scale_identity) for layer, f in enumerate(filters)]))

        # Fully connected layers
        fc = []
        if dropout > 0:
            fc.append(nn.Dropout(p=dropout))
        if n_hidden > 0:
            fc.append(nn.Linear(filters[-1], n_hidden))
            fc.append(nn.ReLU(inplace=True))
            if dropout > 0:
                fc.append(nn.Dropout(p=dropout))
            n_last = n_hidden
        else:
            n_last = filters[-1]
        fc.append(nn.Linear(n_last, out_features))
        self.fc = nn.Sequential(*fc)

    def forward(self, data):
        x = self.gconv(data)[0]
        x = torch.max(x, dim=1)[0].squeeze()  # max pooling over nodes (usually performs better than average)
        x = self.fc(x)
        return x


class GraphUnet(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 filters=[64, 64, 64],
                 K=1,
                 bnorm=False,
                 n_hidden=0,
                 dropout=0.2,
                 adj_sq=False,
                 scale_identity=False,
                 shuffle_nodes=False,
                 visualize=False,
                 pooling_ratios=[0.8, 0.8]):
        super(GraphUnet, self).__init__()

        self.shuffle_nodes = shuffle_nodes
        self.visualize = visualize
        self.pooling_ratios = pooling_ratios
        # Graph convolution layers
        self.gconv = nn.ModuleList([GraphConv(in_features=in_features if layer == 0 else filters[layer - 1],
                                              out_features=f,
                                              K=K,
                                              activation=nn.ReLU(inplace=True),
                                              bnorm=bnorm,
                                              adj_sq=adj_sq,
                                              scale_identity=scale_identity) for layer, f in enumerate(filters)])
        # Pooling layers
        self.proj = []
        for layer, f in enumerate(filters[:-1]):
            # Initialize projection vectors similar to weight/bias initialization in nn.Linear
            fan_in = filters[layer]
            p = Parameter(torch.Tensor(fan_in, 1))
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(p, -bound, bound)
            self.proj.append(p)
        self.proj = nn.ParameterList(self.proj)

        # Fully connected layers
        fc = []
        if dropout > 0:
            fc.append(nn.Dropout(p=dropout))
        if n_hidden > 0:
            fc.append(nn.Linear(filters[-1], n_hidden))
            if dropout > 0:
                fc.append(nn.Dropout(p=dropout))
            n_last = n_hidden
        else:
            n_last = filters[-1]
        fc.append(nn.Linear(n_last, out_features))
        self.fc = nn.Sequential(*fc)

    def forward(self, data):
        # data: [node_features, A, graph_support, N_nodes, label]
        if self.shuffle_nodes:
            # shuffle nodes to make sure that the model does not adapt to nodes order (happens in some cases)
            N = data[0].shape[1]
            idx = torch.randperm(N)
            data = (data[0][:, idx], data[1][:, idx, :][:, :, idx], data[2][:, idx], data[3])

        sample_id_vis, N_nodes_vis = -1, -1
        for layer, gconv in enumerate(self.gconv):
            N_nodes = data[3]

            # TODO: remove dummy or dropped nodes for speeding up forward/backward passes
            # data = (data[0][:, :N_nodes_max], data[1][:, :N_nodes_max, :N_nodes_max], data[2][:, :N_nodes_max], data[3])

            x, A = data[:2]

            B, N, _ = x.shape

            # visualize data
            if self.visualize and layer < len(self.gconv) - 1:
                for b in range(B):
                    if (layer == 0 and N_nodes[b] < 20 and N_nodes[b] > 10) or sample_id_vis > -1:
                        if sample_id_vis > -1 and sample_id_vis != b:
                            continue
                        if N_nodes_vis < 0:
                            N_nodes_vis = N_nodes[b]
                        plt.figure()
                        plt.imshow(A[b][:N_nodes_vis, :N_nodes_vis].data.cpu().numpy())
                        plt.title('layer %d, Input adjacency matrix' % (layer))
                        plt.savefig('input_adjacency_%d.png' % layer)
                        sample_id_vis = b
                        break

            mask = data[2].clone()  # clone as we are going to make inplace changes
            x = gconv(data)[0]  # graph convolution
            if layer < len(self.gconv) - 1:
                B, N, C = x.shape
                y = torch.mm(x.view(B * N, C), self.proj[layer]).view(B, N)  # project features
                y = y / (torch.sum(self.proj[layer] ** 2).view(1, 1) ** 0.5)  # node scores used for ranking below
                idx = torch.sort(y, dim=1)[1]  # get indices of y values in the ascending order
                N_remove = (N_nodes.float() * (1 - self.pooling_ratios[layer])).long()  # number of removed nodes

                # sanity checks
                assert torch.all(
                    N_nodes > N_remove), 'the number of removed nodes must be large than the number of nodes'
                for b in range(B):
                    # check that mask corresponds to the actual (non-dummy) nodes
                    assert torch.sum(mask[b]) == float(N_nodes[b]), (torch.sum(mask[b]), N_nodes[b])

                N_nodes_prev = N_nodes
                N_nodes = N_nodes - N_remove

                for b in range(B):
                    idx_b = idx[b, mask[b, idx[b]] == 1]  # take indices of non-dummy nodes for current data example
                    assert len(idx_b) >= N_nodes[b], (
                        len(idx_b), N_nodes[b])  # number of indices must be at least as the number of nodes
                    mask[b, idx_b[:N_remove[b]]] = 0  # set mask values corresponding to the smallest y-values to 0

                # sanity checks
                for b in range(B):
                    # check that the new mask corresponds to the actual (non-dummy) nodes
                    assert torch.sum(mask[b]) == float(N_nodes[b]), (
                        b, torch.sum(mask[b]), N_nodes[b], N_remove[b], N_nodes_prev[b])
                    # make sure that y-values of selected nodes are larger than of dropped nodes
                    s = torch.sum(y[b] >= torch.min((y * mask.float())[b]))
                    assert s >= float(N_nodes[b]), (s, N_nodes[b], (y * mask.float())[b])

                mask = mask.unsqueeze(2)
                x = x * torch.tanh(y).unsqueeze(2) * mask  # propagate only part of nodes using the mask
                A = mask * A * mask.view(B, 1, N)
                mask = mask.squeeze()
                data = (x, A, mask, N_nodes)

                # visualize data
                if self.visualize and sample_id_vis > -1:
                    b = sample_id_vis
                    plt.figure()
                    plt.imshow(y[b].view(N, 1).expand(N, 2)[:N_nodes_vis].data.cpu().numpy())
                    plt.title('Node ranking')
                    plt.colorbar()
                    plt.savefig('nodes_ranking_%d.png' % layer)
                    plt.figure()
                    plt.imshow(mask[b].view(N, 1).expand(N, 2)[:N_nodes_vis].data.cpu().numpy())
                    plt.title('Pooled nodes (%d/%d)' % (mask[b].sum(), N_nodes_prev[b]))
                    plt.savefig('pooled_nodes_mask_%d.png' % layer)
                    plt.figure()
                    plt.imshow(A[b][:N_nodes_vis, :N_nodes_vis].data.cpu().numpy())
                    plt.title('Pooled adjacency matrix')
                    plt.savefig('pooled_adjacency_%d.png' % layer)
                    print('layer %d: visualizations saved ' % layer)

        if self.visualize and sample_id_vis > -1:
            self.visualize = False  # to prevent visualization for the following batches

        x = torch.max(x, dim=1)[0].squeeze()  # max pooling over nodes
        x = self.fc(x)
        return x


class MGCN(nn.Module):
    '''
    Multigraph Convolutional Network based on (B. Knyazev et al., "Spectral Multigraph Networks for Discovering and Fusing Relationships in Molecules")
    '''

    def __init__(self,
                 in_features,
                 out_features,
                 n_relations,
                 filters=[64, 64, 64],
                 K=1,
                 bnorm=False,
                 n_hidden=0,
                 n_hidden_edge=32,
                 dropout=0.2,
                 adj_sq=False,
                 scale_identity=False):
        super(MGCN, self).__init__()

        # Graph convolution layers
        self.gconv = nn.Sequential(*([GraphConv(in_features=in_features if layer == 0 else filters[layer - 1],
                                                out_features=f,
                                                n_relations=n_relations,
                                                K=K,
                                                activation=nn.ReLU(inplace=True),
                                                bnorm=bnorm,
                                                adj_sq=adj_sq,
                                                scale_identity=scale_identity) for layer, f in enumerate(filters)]))

        # Edge prediction NN
        self.edge_pred = nn.Sequential(nn.Linear(in_features * 2, n_hidden_edge),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(n_hidden_edge, 1))

        # Fully connected layers
        fc = []
        if dropout > 0:
            fc.append(nn.Dropout(p=dropout))
        if n_hidden > 0:
            fc.append(nn.Linear(filters[-1], n_hidden))
            if dropout > 0:
                fc.append(nn.Dropout(p=dropout))
            n_last = n_hidden
        else:
            n_last = filters[-1]
        fc.append(nn.Linear(n_last, out_features))
        self.fc = nn.Sequential(*fc)

    def forward(self, data):
        # data: [node_features, A, graph_support, N_nodes, label]

        # Predict edges based on features
        x = data[0]
        B, N, C = x.shape
        mask = data[2]
        # find indices of nodes
        x_cat, idx = [], []
        for b in range(B):
            n = int(mask[b].sum())
            node_i = torch.nonzero(mask[b]).repeat(1, n).view(-1, 1)
            node_j = torch.nonzero(mask[b]).repeat(n, 1).view(-1, 1)
            triu = (node_i < node_j).squeeze()  # skip loops and symmetric connections
            x_cat.append(torch.cat((x[b, node_i[triu]], x[b, node_j[triu]]), 2).view(int(torch.sum(triu)), C * 2))
            idx.append((node_i * N + node_j)[triu].squeeze())

        x_cat = torch.cat(x_cat)
        idx_flip = np.concatenate((np.arange(C, 2 * C), np.arange(C)))
        # predict values and encourage invariance to nodes order
        y = torch.exp(0.5 * (self.edge_pred(x_cat) + self.edge_pred(x_cat[:, idx_flip])).squeeze())
        A_pred = torch.zeros(B, N * N, device=cuda_device)
        c = 0
        for b in range(B):
            A_pred[b, idx[b]] = y[c:c + idx[b].nelement()]
            c += idx[b].nelement()
        A_pred = A_pred.view(B, N, N)
        A_pred = (A_pred + A_pred.permute(0, 2, 1))  # assume undirected edges

        # Use both annotated and predicted adjacency matrices to learn a GCN
        data = (x, torch.stack((data[1], A_pred), 3), mask)
        x = self.gconv(data)[0]
        x = torch.max(x, dim=1)[0].squeeze()  # max pooling over nodes
        x = self.fc(x)
        return x
