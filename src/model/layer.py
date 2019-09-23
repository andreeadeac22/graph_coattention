import torch.nn as nn

class DNNBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(DNNBlock, self).__init__()
        self.lin = nn.Linear(in_features, out_features)
        self.drop = nn.Dropout(p=dropout)
        self.actv = nn.LeakyReLU()

    def forward(self, x):
        x = self.lin(x)
        x = self.drop(x)
        x = self.actv(x)
        return x