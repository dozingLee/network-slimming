import os
import argparse
import numpy as np
from models import *
import matplotlib.pyplot as plt
import models

''' 
    e.g. vgg19-cifar10 pruning rates (0.5, 0.7) with y-limit
    python vggprune_show.py --arch vgg --dataset cifar10 --depth 19 -y-limit --pruning-rates 0.5 0.7
        --model logs/sparsity_vgg19_cifar10_s_1e_4/model_best.pth.tar  --save logs/sparsity_vgg19_cifar10_s_1e_4
    
    e.g. pruned model
    python vggprune_show.py --arch vgg --dataset cifar100 --depth 19
        --model logs/prune_vgg19_cifar100_percent_0.5/pruned.pth.tar  --save logs/prune_vgg19_cifar100_percent_0.5
    
    e.g. fine-tining model
'''

# Prune settings
parser = argparse.ArgumentParser(description='Model Weight Visualization')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--arch', default='vgg', type=str,
                    help='architecture to use (vgg, resnet, densenet)')
parser.add_argument('--depth', type=int, default=19,
                    help='depth of the neural network (default: 19)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('--pruning-rates', default=[], type=float, metavar='R', nargs='*',
                    help='pruning rates list (default: none)')
parser.add_argument('-y-limit', action='store_true', default=False,
                    help='y limit at [0, 1] for the global view (default: False)')

args = parser.parse_args()


def get_channel_variation(list_1d, list_2d, pruning_rate):
    """
    :param list_1d: data 1 dimension
    :param list_2d: data 2 dimension
    :param pruning_rate: pruning rate
    :return threshold, channel remaining
    """
    y = np.sort(list_1d)
    thre_index = int(len(list_1d) * pruning_rate)
    threshold = y[thre_index]
    channel_remain = []
    for data in list_2d:
        mask = data > threshold
        channel_remain.append(np.sum(mask))
    return threshold, channel_remain


def plot(list1d, list2d, channel_origin, pruning_rates, title, y_limit):
    """
    :param list1d: data 1 dimension
    :param list2d: data 2 dimension
    :param channel_origin: origin channel list
    :param pruning_rates: e.g. [0.5, 0.7]
    :param title: plot title
    :param y_limit: if True: y[0, 1] ,else: y no limit
    """
    # Data: transfer data into list
    data_list, index_list, tick_list = [], [], []
    start = 0
    num_layer = len(list2d)
    num_total = len(list1d)
    for i in range(num_layer):
        length = len(list2d[i])
        data_list.append(list(bn_2d_list[i]))
        index_list.append(list(range(start, start + length)))
        start += length
        tick_list.append(start)

    # Settings
    colors = [plt.cm.tab10(i / float(num_layer - 1)) for i in range(num_layer)]
    plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
    point_size = 6
    font_size = 8
    title_font_size = 14

    # Plot threshold line
    record_thre, record_vary, record_linex, record_liney = [], [], [], []
    num_plot = 1
    if pruning_rates:
        for pruning_rate in pruning_rates:
            thre, vary = get_channel_variation(list1d, list2d, pruning_rate)
            line_x = np.linspace(0, num_total, num_total * 20)
            line_y = np.ones_like(line_x) * thre
            record_thre.append(thre)
            record_vary.append(vary)
            record_linex.append(line_x)
            record_liney.append(line_y)
        num_plot += len(pruning_rates)

    # Plot global graph
    plt.subplot(num_plot, 1, 1)
    for i, item in enumerate(data_list):
        plt.scatter(x=index_list[i], y=data_list[i], s=point_size, c=colors[i], label=str(i))
    for i, item in enumerate(record_linex):
        plt.plot(record_linex[i], record_liney[i], label='rate {}'.format(pruning_rates[i]))
    if y_limit:
        y_lim_l, y_lim_r = 0.0, 1.0
        plt.gca().set(ylim=(y_lim_l, y_lim_r), ylabel='Weight Data')
    plt.xticks(tick_list, fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.title("The {} Weight Overview".format(title), fontsize=title_font_size)
    plt.legend(fontsize=font_size, bbox_to_anchor=(1.02, 0), loc=3, borderaxespad=0)

    # Plot threshold graph
    threshold_loc = 1.5
    if pruning_rates:
        for i, pruning_rate in enumerate(pruning_rates):
            plt.subplot(num_plot, 1, i + 2)
            y_lim_l, y_lim_r = 0.0, record_thre[i] * threshold_loc
            labels = ["{:>3d} - {:>3d}".format(item, record_vary[i][j])
                      for j, item in enumerate(channel_origin)]
            for k, data_item in enumerate(data_list):
                plt.scatter(x=index_list[k], y=data_list[k], s=point_size, c=colors[k], label=labels[k])
            plt.gca().set(ylim=(y_lim_l, y_lim_r), ylabel='Weight Data')
            plt.xticks(tick_list, fontsize=font_size)
            plt.yticks(fontsize=font_size)
            plt.plot(record_linex[i], record_liney[i], label='rate {}'.format(pruning_rate))
            plt.title("The {} Weight Pruning Rate {}".format(title, pruning_rate), fontsize=title_font_size)
            plt.legend(fontsize=font_size, bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0)

    plt.savefig(os.path.join(args.save, "model weight.png"))
    plt.show()


if __name__ == '__main__':
    # ======= Preprocess ========
    # folder
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    # model
    best_prec1 = 0.
    if args.model:
        if os.path.isfile(args.model):
            print("=> loading checkpoint '{}'".format(args.model))
            model_file = torch.load(args.model)
            if 'cfg' in model_file:
                model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=model_file['cfg'])
            else:
                model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
            model.load_state_dict(model_file['state_dict'])
            if 'best_prec1' in model_file:
                best_prec1 = model_file['best_prec1']
            print("=> loaded checkpoint '{}'".format(args.model))
        else:
            raise Exception("=> no model file found at '{}'".format(args.model))
    print(model)

    # ======= Handle Data ========
    # BN 2d weight data
    #   num_total: number of BatchNorm2d weight
    #   num_layer: number of BatchNorm2d layer
    #   bn_2d_list: record BatchNorm2d weight
    #   channel_list: record BatchNorm2d length
    num_total = 0
    bn_2d_list, channel_list = [], []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            length = m.weight.data.shape[0]
            num_total += length
            bn_2d_list.append(m.weight.data.cpu().numpy())
            channel_list.append(length)

    # BN 1d weight data
    #   bn_1d_list: record BatchNorm 1d weight
    bn_1d_list = torch.zeros(num_total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn_1d_list[index:(index + size)] = m.weight.data.abs().clone()
            index += size

    # ======= Visualization ========
    if best_prec1 > 0.:
        model_name = '{}{}-{} Model (best prec: {})'.format(args.arch, args.depth, args.dataset, best_prec1)
    else:
        model_name = '{}{}-{} Model'.format(args.arch, args.depth, args.dataset)
    plot(bn_1d_list, bn_2d_list, channel_list, args.pruning_rates, model_name, args.y_limit)
