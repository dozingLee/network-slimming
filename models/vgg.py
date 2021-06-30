import math
import torch
import torch.nn as nn
from torch.autograd import Variable

__all__ = ['vgg']

default_cfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class vgg(nn.Module):
    def __init__(self, dataset='cifar10', depth=19, cfg=None, conv_cfg=None):
        """
        :param dataset: `cifar10` or `cifar100`
        :param depth: `11`, `13`, '16', or `19` (default)
        :param cfg: vgg model convolutional layer's channel config
        :param conv_cfg:
            return convolutional layer's channel config (index starts at 1)
            like [2, 4, 8, 12]
        """
        super(vgg, self).__init__()

        # model value
        self.conv_cfg = conv_cfg

        # model config
        if cfg is None:
            cfg = default_cfg[depth]

        # model feature
        self.feature = make_layers(cfg)

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

        # model classifier
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)

        # return value
        if len(conv_value):
            return y, conv_value
        return y


if __name__ == '__main__':
    net = vgg(conv_cfg=[2, 4, 8, 12])
    input = Variable(torch.FloatTensor(64, 3, 32, 32))
    output, value = net(input)
    print('y.data.shape: {}, value length: {}'.format(input.data.shape, len(value)))
