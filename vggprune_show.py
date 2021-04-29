import os
import argparse
from models import *
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings(action='once')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

'''
    vggprune_show.py: VGG sparsity model's weight data show
        threshold rate: just use --percent 0.7 to replace --percent 0.5
    
    (1) Baseline
    Baseline vgg19 for cifar10
    > python vggprune_show.py --dataset cifar10 --depth 19 --percent 0.5
        --model ./logs/baseline_vgg19_cifar10/model_best.pth.tar --save ./logs/baseline_vgg19_cifar10
    
    Baseline vgg19 for cifar100
    > python vggprune_show.py --dataset cifar100 --depth 19 --percent 0.7
        --model ./logs/baseline_vgg19_cifar100/model_best.pth.tar --save ./logs/baseline_vgg19_cifar100
    
    (2) Sparsity
    VGG19 with sparsity 1e-4 and threshold rate 0.5 for cifar10
    > python vggprune_show.py --dataset cifar10 --depth 19 --percent 0.5
        --model ./logs/sparsity_vgg19_cifar10_s_1e-4/model_best.pth.tar --save ./logs/sparsity_vgg19_cifar10_s_1e-4
    
    VGG19 with sparsity 1e-4 and threshold rate 0.7 for cifar10
    > python vggprune_show.py --dataset cifar10 --depth 19 --percent 0.7
        --model ./logs/sparsity_vgg19_cifar10_s_1e-4/model_best.pth.tar --save ./logs/sparsity_vgg19_cifar10_s_1e-4
    
     
'''

default_cfg = {
    11: [64, 128, 256, 256, 512, 512, 512, 512],
    13: [64, 64, 128, 128, 256, 256, 512, 512, 512, 512],
    16: [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512],
    19: [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512],
}

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

# BN 2d weight data
num_total = 0
bn_2d_list = []
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        num_total += m.weight.data.shape[0]
        bn_2d_list.append(m.weight.data.cpu().numpy())
num_layer = len(bn_2d_list)

# BN 1d weight data
bn_1d_list = torch.zeros(num_total)
index = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn_1d_list[index:(index + size)] = m.weight.data.abs().clone()
        index += size

# BN 1d weight data threshold
y, i = torch.sort(bn_1d_list)  # descending order, y: sort list, i: index list
threshold_index = int(num_total * args.percent)  # threshold index
threshold = y[threshold_index]  # threshold value
bn_1d_list_sort = y.data.cpu().numpy()
bn_1d_index_list_sort = i.data.cpu().numpy()

# BN 2d layers
channel_origin = default_cfg[args.depth]
channel_prune = []
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        weight_copy = m.weight.data.abs().clone()
        mask = weight_copy.gt(threshold).float().cuda()
        channel_prune.append(int(torch.sum(mask)))
channel_variation = list(np.array(channel_origin) - np.array(channel_prune))

# bn_rank_2d_list = [[] for i in range(num_layer)]
# 不要使用[] * n，这是浅拷贝，即一旦array改变，matrix中3个list也会随之改变

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
point_size = 6
font_size = 8
title_font_size = 14

# Threshold line
line_x = np.linspace(0, num_total, num_total * 20)
line_y = np.ones_like(line_x) * threshold.cpu().numpy()
plt.plot(line_x, line_y, label='threshold')
threshold_above_rate = 5.0
threshold_below_rate = 1.1

# Legend decorations 1
plt.subplot(3, 1, 1)
for i, item in enumerate(scatter_list):
    plt.scatter(x=scatter_index_list[i], y=scatter_list[i],
                s=point_size, c=colors[i], label=str(i))
y_lim_l, y_lim_r = 0.0, 1.0
plt.gca().set(ylim=(y_lim_l, y_lim_r), ylabel='Weight Data')
plt.xticks(range(0, num_total, int(num_total / num_layer)), fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.title("Scatter Weight Data({}~{:.2e}), Threshold({:.2e}) and Percent({})"
          .format(y_lim_l, y_lim_r, threshold, args.percent), fontsize=title_font_size)
plt.legend(fontsize=font_size, bbox_to_anchor=(1.02, 0), loc=3, borderaxespad=0)

# Legend decorations 2
plt.subplot(3, 1, 2)
y_lim_l, y_lim_r = 0.0, threshold * threshold_above_rate
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
y_lim_l, y_lim_r = 0.0, threshold * threshold_below_rate
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

