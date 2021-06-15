import math
import torch
import torch.nn as nn
from torch.autograd import Variable

__all__ = ['vgg']

defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class vgg(nn.Module):
    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None, batch_norm=True, conv_cfg=None):
        super(vgg, self).__init__()

        self.conv_cfg = conv_cfg
        if cfg is None:
            cfg = defaultcfg[depth]

        self.feature = self.make_layers(cfg, batch_norm)

        num_classes = 10
        if dataset == 'cifar100':
            num_classes = 100
        self.classifier = nn.Linear(cfg[-1], num_classes)

        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        conv_value = []
        if self.conv_cfg:
            conv_idx = 0
            for k, m in enumerate(self.feature):
                x = m(x)
                if isinstance(m, nn.Conv2d):
                    conv_idx += 1
                    if conv_idx in self.conv_cfg:
                        conv_value.append(x)
        else:
            x = self.feature(x)

        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)

        if len(conv_value):
            return y, conv_value
        else:
            return y

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


if __name__ == '__main__':
    net = vgg(conv_cfg=[2, 4, 8, 12])
    x = Variable(torch.FloatTensor(16, 3, 40, 40))
    y, value = net(x)
    print(y.data.shape, len(value))
