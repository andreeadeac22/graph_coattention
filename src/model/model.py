import torch
import torch.nn as nn

import layer

class GraphPairNN(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim):
        super(GraphPairNN, self).__init__()
        self.out_features = out_features
        self.layer1 = layer.DNNBlock(in_features, hidden_dim, 0.5).cuda()
        self.layer2 = layer.DNNBlock(hidden_dim, hidden_dim, 0.5).cuda()
        self.layer3 = [layer.DNNBlock(hidden_dim, out_features, 0.3).cuda() for _ in range(0,self.out_features)]

    def forward(self, x):
        z = self.layer1(x)
        z = torch.stack([self.layer3[i](z) for i in range(0,self.out_features)], dim = 1)
        return z


