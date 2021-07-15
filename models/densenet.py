import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .channel_selection import channel_selection

__all__ = ['densenet']

"""
densenet with basic block.
"""


class BasicBlock(nn.Module):
    def __init__(self, in_planes, cfg, growth_rate=12, drop_rate=0):
        """
        :param in_planes: input channel size
        :param cfg: `in_planes` equals `cfg`
        :param growth_rate: output channel size = `in_planes` + growth_rate
        :param drop_rate: dropout rate
        """
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.select = channel_selection(in_planes)
        self.conv1 = nn.Conv2d(cfg, growth_rate, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = drop_rate

    def forward(self, x):
        out = self.bn1(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)
        out = torch.cat((x, out), 1)
        return out

    def forward_bn(self, x):
        out = self.bn1(x)
        bn_value = out.clone()
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)
        out = torch.cat((x, out), 1)
        return out, bn_value


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes, cfg):
        """
        :param in_planes: number of the input channel
        :param out_planes: number of the output channel
        :param cfg: `out_planes` equals `cfg`
        """
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.select = channel_selection(in_planes)
        self.conv1 = nn.Conv2d(cfg, out_planes, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out

    def forward_bn(self, x):
        out = self.bn1(x)
        bn_value = out.clone()
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out, bn_value


class densenet(nn.Module):
    def __init__(self, depth=40, drop_rate=0, dataset='cifar10', growth_rate=12, compression_rate=1, cfg=None):
        super(densenet, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) // 3
        block = BasicBlock

        self.growth_rate = growth_rate
        self.drop_rate = drop_rate

        if cfg is None:
            cfg = []
            start = growth_rate * 2
            for _ in range(3):
                cfg.append([start + growth_rate * i for i in range(n + 1)])
                start += growth_rate * n
            cfg = [item for sub_list in cfg for item in sub_list]
        assert len(cfg) == 3 * n + 3, 'length of config variable cfg should be 3n+3'

        # self.in_planes is a global variable used across multiple
        self.in_planes = growth_rate * 2
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_dense_block(block, n, cfg[0:n])
        self.trans1 = self._make_transition(compression_rate, cfg[n])
        self.dense2 = self._make_dense_block(block, n, cfg[n + 1:2 * n + 1])
        self.trans2 = self._make_transition(compression_rate, cfg[2 * n + 1])
        self.dense3 = self._make_dense_block(block, n, cfg[2 * n + 2:3 * n + 2])
        self.bn = nn.BatchNorm2d(self.in_planes)
        self.select = channel_selection(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        else:
            raise ValueError('Model `dataset` parameter is Error!')
        self.classifier = nn.Linear(cfg[-1], num_classes)

        # Weight initialization
        self._initialize_weights()

    def _make_dense_block(self, block, num_block, cfg):
        """
        :param block: Basic Block (one block means one conv2d)
        :param num_block: number of blocks (n) <- in every layer
        :param cfg: channel config of all blocks <- in every layer
        """
        layers = []
        assert num_block == len(cfg), 'Length of the cfg parameter is not right.'
        for i in range(num_block):
            layers.append(block(self.in_planes, cfg=cfg[i], growth_rate=self.growth_rate, drop_rate=self.drop_rate))
            self.in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def _make_transition(self, compression_rate, cfg):
        """
        :param compression_rate: compress input channel
        :param cfg: input channel size, `cfg` equals `in_planes`
        """
        # cfg is a number in this case.
        in_planes = self.in_planes
        out_planes = int(math.floor(self.in_planes // compression_rate))
        self.in_planes = out_planes
        return Transition(in_planes, out_planes, cfg)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)

        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.dense3(x)
        x = self.bn(x)
        x = self.select(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

