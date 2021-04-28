import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import *
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='once')

'''
    vggprune_show.py: VGG sparsity model's weight data show
    
    (1) VGG shows 50% proportion with sparsity 1e-4 for cifar10
    > python vggprune_show.py --dataset cifar10 --depth 19 --percent 0.5
        --model ./logs/sparsity_vgg19_cifar10_s_1e-4/model_best.pth.tar --save ./logs/sparsity_vgg19_cifar10_s_1e-4
    
    (2) VGG shows 70% proportion with sparsity 1e-4 for cifar10
    > python vggprune_show.py --dataset cifar10 --depth 19 --percent 0.7
        --model ./logs/sparsity_vgg19_cifar10_s_1e-4/model_best.pth.tar --save ./logs/sparsity_vgg19_cifar10_s_1e-4
    
'''

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=19,
                    help='depth of the vgg')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if not os.path.exists(args.save):
    os.makedirs(args.save)

# Model
model = vgg(dataset=args.dataset, depth=args.depth)
if args.cuda:
    model.cuda()
if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.model))
print(model)

# model batch normalization weight data
num_total = 0
bn_2d_list = []
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        num_total += m.weight.data.shape[0]
        bn_2d_list.append(m.weight.data.cpu().numpy())
num_layer = len(bn_2d_list)

bn_1d_list = torch.zeros(num_total)
index = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn_1d_list[index:(index+size)] = m.weight.data.abs().clone()
        index += size

# weight data threshold
y, i = torch.sort(bn_1d_list)  # descending order, y: sort list, i: index list
threshold_index = int(num_total * args.percent)  # threshold index
threshold = y[threshold_index]  # threshold value

bn_1d_list_sort = y.data.cpu().numpy()
bn_1d_index_list_sort = i.data.cpu().numpy()

bn_rank_2d_list = [[] for i in range(num_layer)]  # 不要使用[] * n，这是浅拷贝，即一旦array改变，matrix中3个list也会随之改变

# Plot1: every layer's index & data
scatter_list = []
scatter_index_list = []
start = 0
for i in range(num_layer):
    length = len(bn_2d_list[i])
    scatter_list.append(list(bn_2d_list[i]))
    scatter_index_list.append(list(range(start, start + length)))
    start += length

# Plot settings
colors = [plt.cm.tab10(i / float(num_layer - 1)) for i in range(num_layer)]
plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
point_size = 8
font_size = 10
title_font_size = 16


plt.subplot(3, 1, 1)

# Weight data point
for i, item in enumerate(scatter_list):
    plt.scatter(x=scatter_index_list[i], y=scatter_list[i],
                s=point_size, c=colors[i], label=str(i))

# Threshold line
line_x = np.linspace(0, num_total, num_total * 20)
line_y = np.ones_like(line_x) * threshold.cpu().numpy()
plt.plot(line_x, line_y, label='threshold')

# Legend decorations 1
y_lim_l, y_lim_r = 0.0, 1.0
plt.gca().set(ylim=(y_lim_l, y_lim_r), ylabel='Weight Data')
plt.xticks(range(0, num_total, int(num_total/num_layer)), fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.title("Scatter Weight Data({}~{:.2e}), Threshold({:.2e}) and Percent({})"
          .format(y_lim_l, y_lim_r, threshold, args.percent), fontsize=title_font_size)
plt.legend(fontsize=font_size, bbox_to_anchor=(1.02, 0), loc=3, borderaxespad=0)


# Legend decorations 2
plt.subplot(3, 1, 2)
threshold_rate = 5.0
y_lim_l, y_lim_r = 0.0, threshold * threshold_rate
channel_origin = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]  # vgg19 model
channel_prune = [35, 64, 128, 128, 255, 251, 222, 180, 113, 46, 44, 34, 17, 21, 33, 80]  # prune.txt
channel_variation = list(np.array(channel_origin) - np.array(channel_prune))
labels = ["{:>3d} - {:>3d}".format(origin_item, channel_prune[i]) for i, origin_item in enumerate(channel_origin)]
for i, bn_item in enumerate(scatter_list):
    plt.scatter(x=scatter_index_list[i], y=scatter_list[i],
                s=point_size, c=colors[i], label=labels[i])
plt.title("Scatter Weight Data({}~{:.2e}) and Threshold({:.2e}) and Percent({}) "
          .format(y_lim_l, y_lim_r, threshold, args.percent), fontsize=title_font_size)
plt.plot(line_x, line_y, label='thres above')
plt.gca().set(ylim=(y_lim_l, y_lim_r), ylabel='Weight Data')
plt.legend(fontsize=font_size, bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0)


# Legend decorations 3
plt.subplot(3, 1, 3)
threshold_rate = 1.1
y_lim_l, y_lim_r = 0.0, threshold * threshold_rate
labels = ["{:>3d} - {:>3d}".format(origin_item, channel_variation[i]) for i, origin_item in enumerate(channel_origin)]
for i, bn_item in enumerate(scatter_list):
    plt.scatter(x=scatter_index_list[i], y=scatter_list[i],
                s=point_size, c=colors[i], label=labels[i])
plt.title("Scatter Weight Data({}~{:.2e}) and Threshold({:.2e}) and Percent({}) "
          .format(y_lim_l, y_lim_r, threshold, args.percent), fontsize=title_font_size)
plt.plot(line_x, line_y, label='thres below')
plt.gca().set(ylim=(y_lim_l, y_lim_r), ylabel='Weight Data')
plt.legend(fontsize=font_size, bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0)

plt.savefig(os.path.join(args.save, "vggprune_show_percent_{}.png".format(args.percent)))
plt.show()

# def encoder(data_i):
#     for i, index_i in enumerate(bn_index_list):
#         if data_i <= index_i:
#             return i
#
# for i, data in enumerate(bn_sort_list):
#     j = encoder(bn_sort_index_list[i])
#     bn_rank_list[j].append(data)