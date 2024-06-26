from torch import nn
import torch.nn.functional as F

from .conv import Conv1d, Conv2d
from .linear import FC


class MLP(nn.ModuleList):
    def __init__(self,
                 in_channels,
                 mlp_channels,
                 dropout=None,
                 bn=True,
                 bn_momentum=0.1):
        super(MLP, self).__init__()

        self.in_channels = in_channels
        self.dropout = dropout

        for ind, out_channels in enumerate(mlp_channels):
            self.append(FC(in_channels, out_channels,
                           relu=True, bn=bn, bn_momentum=bn_momentum))
            in_channels = out_channels

        self.out_channels = in_channels

    def forward(self, x):
        for module in self:
            x = module(x)
            if self.dropout:
                x = F.dropout(x, self.dropout, self.training, inplace=False)
        return x


class SharedMLP(nn.ModuleList):
    def __init__(self,
                 in_channels,
                 mlp_channels,
                 ndim=1,
                 bn=True,
                 bn_momentum=0.1):
        super(SharedMLP, self).__init__()

        self.in_channels = in_channels

        if ndim == 1:
            mlp_module = Conv1d
        elif ndim == 2:
            mlp_module = Conv2d
        else:
            raise ValueError()

        for ind, out_channels in enumerate(mlp_channels):
            self.append(mlp_module(in_channels, out_channels, 1,
                                   relu=True, bn=bn, bn_momentum=bn_momentum))
            in_channels = out_channels

        self.out_channels = in_channels

    def forward(self, x):
        for module in self:
            x = module(x)
        return x
