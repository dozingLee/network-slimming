from __future__ import absolute_import
import math
import torch.nn as nn
from .channel_selection import channel_selection

__all__ = ['resnet']

"""
preactivation resnet with bottleneck design.
"""


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg, stride=1, down_sample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.select = channel_selection(inplanes)
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x

        # group1
        out = self.bn1(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)

        # group2
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        # group3
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        # down sample
        if self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        return out

    def mask(self, index, cfg_mask):
        if index == 0:
            self.bn1.weight.data.mul_(cfg_mask)
            self.bn1.bias.data.mul_(cfg_mask)
        elif index == 1:
            self.bn2.weight.data.mul_(cfg_mask)
            self.bn2.bias.data.mul_(cfg_mask)
        elif index == 2:
            self.bn3.weight.data.mul_(cfg_mask)
            self.bn3.bias.data.mul_(cfg_mask)
        else:
            raise ValueError("Index is not including.")

    def forward_bn(self, x):
        bn_value = []
        residual = x

        # group1
        out = self.bn1(x)
        bn_value.append(out.clone())
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)

        # group2
        out = self.bn2(out)
        bn_value.append(out.clone())
        out = self.relu(out)
        out = self.conv2(out)

        # group3
        out = self.bn3(out)
        bn_value.append(out.clone())
        out = self.relu(out)
        out = self.conv3(out)

        # down sample
        if self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        return out, bn_value


class resnet(nn.Module):
    def __init__(self, depth=164, dataset='cifar10', cfg=None, block_cfg=None):
        """
        :param depth:
            164 layers => 1 conv2d + 3 layers × 18 blocks (every layer)  × 3 conv2ds (every block)  + 1 avgPool2d
            param n = (depth - 2) // 9:
                n means how many blocks in every layer
                9 = 3 layers × 3 conv2d (every block)
        :param cfg:
            if depth = 164, then len(cfg) = 164
        :param conv_cfg:
            every layer block indexes: [6, 12, 18], [9, 18], [18]
            every `conv_cfg` value should <= `n` (if depth = 164, n = 18)

        number of BatchNorm2d:
            163 = 162 (3 layers × 18 Bottlenecks × 3 BatchNorm2ds) + 1 BatchNorm2d
        """
        super(resnet, self).__init__()
        assert (depth - 2) % 9 == 0, 'depth should be 9n+2'

        # model value
        n = (depth - 2) // 9  # depth = 164, n = 18
        block = Bottleneck
        self.block_cfg = block_cfg
        self.inplanes = 16

        # model config
        if cfg is None:
            cfg = [[16, 16, 16], [64, 16, 16] * (n - 1),
                   [64, 32, 32], [128, 32, 32] * (n - 1),
                   [128, 64, 64], [256, 64, 64] * (n - 1), [256]]
            cfg = [item for sub_list in cfg for item in sub_list]

        # model feature
        conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        layer1 = self._make_layer(block, 16, n, cfg=cfg[0:3 * n])                # 32 × 32
        layer2 = self._make_layer(block, 32, n, cfg=cfg[3 * n:6 * n], stride=2)  # 16 × 16
        layer3 = self._make_layer(block, 64, n, cfg=cfg[6 * n:9 * n], stride=2)  # 8 × 8
        bn = nn.BatchNorm2d(64 * block.expansion)
        select = channel_selection(64 * block.expansion)
        relu = nn.ReLU(inplace=True)

        feature = [conv1, layer1, layer2, layer3, bn, select, relu]
        self.feature = nn.Sequential(*feature)

        # model classifier
        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        else:
            raise ValueError('Model `dataset` parameter is Error!')
        self.classifier = nn.Linear(cfg[-1], num_classes)

        # model initialize weight
        self._initialize_weights()

    def _make_layer(self, block, planes, num_block, cfg, stride=1):
        """
        :param block: Bottleneck item
        :param planes: record the layer's output channel size
        :param num_block: how many blocks in every layer
        :param cfg:
        """
        down_sample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_sample = nn.Sequential(nn.Conv2d(
                self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False))

        layers = [block(self.inplanes, planes, cfg[0:3], stride, down_sample)]
        self.inplanes = planes * block.expansion
        for i in range(1, num_block):
            layers.append(block(self.inplanes, planes, cfg[3 * i: 3 * (i + 1)]))
        return nn.Sequential(*layers)

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
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        # model feature
        block_value = []
        if self.block_cfg:
            for k, m in enumerate(self.feature):
                if isinstance(m, nn.Sequential):
                    block_idx = 0
                    for j, block_item in enumerate(m):
                        block_idx += 1
                        x = block_item(x)
                        if block_idx in self.block_cfg:
                            block_value.append(x)
                else:
                    x = m(x)
        else:
            x = self.feature(x)

        # model classifier
        x = nn.AvgPool2d(8)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)

        if len(block_value):
            return y, block_value
        return y
