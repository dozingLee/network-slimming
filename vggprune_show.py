import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import *
import matplotlib.pyplot as plt
import warnings;

warnings.filterwarnings(action='once')

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar100)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=19,
                    help='depth of the vgg')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='./logs/model_best_vggnet_sr_93.78.pth.tar', type=str, metavar='PATH',
                    help='path to the model (default: none)')  # ./logs/model_best_vggnet_sr_93.78.pth.tar
parser.add_argument('--save', default='./figure', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

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

# TODO: for i in


bn_list = []
# Number of BatchNorm2d.weight
total = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        total += m.weight.data.shape[0]
        bn_list.append(m.weight.data.cpu().numpy())
len_layer = len(bn_list)


# Array of BatchNorm2d.weight
bn_index_list = []
bn = torch.zeros(total)
index = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn[index:(index + size)] = m.weight.data.abs().clone()
        index += size
        bn_index_list.append(index)


# Threshold
y, i = torch.sort(bn)  # descending order, y: sort list, i: index list
thre_index = int(total * args.percent)  # threshold index
thre = y[thre_index]  # threshold value

bn_sort_list = y.data.cpu().numpy()
bn_sort_index_list = i.data.cpu().numpy()

bn_rank_list = [[] for i in range(len_layer)]  # 不要使用[] * n，这是浅拷贝，即一旦array改变，matrix中3个list也会随之改变

def encoder(data_i):
    for i, index_i in enumerate(bn_index_list):
        if data_i <= index_i:
            return i

for i, data in enumerate(bn_sort_list):
    j = encoder(bn_sort_index_list[i])
    bn_rank_list[j].append(data)

bn_scatter_rank_list = []
start = 0
for i in range(len_layer):
    length = len(bn_rank_list[i])
    bn_scatter_rank_item = {'index': list(range(start, start + length)), 'value': list(bn_rank_list[i])}
    bn_scatter_rank_list.append(bn_scatter_rank_item)
    start += length


bn_scatter_list = []
for i in range(len_layer):
    length = len(bn_list[i])
    bn_scatter_item = {'index': list(range(start, start + length)), 'value': list(bn_list[i])}
    bn_scatter_list.append(bn_scatter_item)
    start += length

colors = [plt.cm.tab10(i / float(len_layer - 1)) for i in range(len_layer)]

# Draw Plot for Each Category
plt.figure(figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
point_size = 8
font_size = 10


plt.subplot(3, 1, 1)

for i, bn_item in enumerate(bn_scatter_list):       # weight data point
    plt.scatter(x=bn_scatter_list[i]['index'], y=bn_scatter_list[i]['value'],
                s=point_size, c=colors[i], label=str(i))
line_x = np.linspace(0, total, total*20)        # threshold line
line_y = np.ones_like(line_x) * thre.cpu().numpy()
plt.plot(line_x, line_y, label='threshold')

# Decorations
y_lim_l, y_lim_r = 0.0, 1.0
plt.gca().set(ylim=(y_lim_l, y_lim_r), xlabel='index', ylabel='weight data')
plt.xticks(range(0, total, int(total/len_layer)), fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.title("Scatterplot of Weight Data({}~{}) and Threshold({:.2e})".format(y_lim_l, y_lim_r, thre), fontsize=22)
plt.legend(fontsize=font_size, bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)


plt.subplot(3, 1, 2)

y_lim_l, y_lim_r = 0.0, 1e-6
channel_origin = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]  # vgg19 model
channel_prune = [35, 64, 128, 128, 255, 251, 222, 180, 113, 46, 44, 34, 17, 21, 33, 80]  # prune.txt
channel_variation = np.array(channel_origin) - np.array(channel_prune)
channel_variation = list(channel_variation)
labels = ["{:>3d}-{:>3d}={:>3d}".format(origin_item, channel_variation[i], channel_prune[i])
          for i, origin_item in enumerate(channel_origin)]

for i, bn_item in enumerate(bn_scatter_list):
    plt.scatter(x=bn_scatter_list[i]['index'], y=bn_scatter_list[i]['value'],
                s=point_size, c=colors[i], label=labels[i])
plt.title("Scatterplot of Weight Data({}~{}) and Threshold({:.2e}) ".format(y_lim_l, y_lim_r, thre), fontsize=22)
plt.plot(line_x, line_y, label='threshold')
plt.gca().set(ylim=(y_lim_l, y_lim_r), xlabel='index', ylabel='weight data')
# plt.legend(fontsize=font_size, loc='upper right')
plt.legend(fontsize=font_size, bbox_to_anchor=(1.03, 0), loc=3, borderaxespad=0)


plt.subplot(3, 1, 3)

for i, bn_item in enumerate(bn_scatter_rank_list):
    plt.scatter(x=bn_scatter_rank_list[i]['index'], y=bn_scatter_rank_list[i]['value'],
                s=point_size, c=colors[i], label=labels[i])
plt.title("Scatterplot of Weight Data({}~{}) and Threshold({:.2e}) ".format(y_lim_l, y_lim_r, thre), fontsize=22)
plt.plot(line_x, line_y, label='threshold')
plt.gca().set(ylim=(y_lim_l, y_lim_r), xlabel='index', ylabel='weight data')
# plt.legend(fontsize=font_size, loc='upper right')
plt.legend(fontsize=font_size, bbox_to_anchor=(1.03, 0), loc=3, borderaxespad=0)

plt.show()

#
# # Pruned
# pruned = 0
# cfg = []
# cfg_mask = []
# for k, m in enumerate(model.modules()):
#     if isinstance(m, nn.BatchNorm2d):
#         weight_copy = m.weight.data.abs().clone()
#         mask = weight_copy.gt(thre).float().cuda()  # if value > threshold, True, else False
#         pruned = pruned + mask.shape[0] - torch.sum(mask)  # Num of pruned
#         m.weight.data.mul_(mask)  # pruned weight
#         m.bias.data.mul_(mask)    # pruned bias
#         cfg.append(int(torch.sum(mask)))
#         cfg_mask.append(mask.clone())
#         print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
#             format(k, mask.shape[0], int(torch.sum(mask))))
#     elif isinstance(m, nn.MaxPool2d):
#         cfg.append('M')
#
# pruned_ratio = pruned/total
#
# print('Pre-processing Successful! Pruned ratio: ', pruned_ratio)
#
#
# # simple test model after Pre-processing prune (simple set BN scales to zeros)
# def test(model):
#     kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
#     if args.dataset == 'cifar10':
#         test_loader = torch.utils.data.DataLoader(
#             datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
#             batch_size=args.test_batch_size, shuffle=True, **kwargs)
#     elif args.dataset == 'cifar100':
#         test_loader = torch.utils.data.DataLoader(
#             datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
#             batch_size=args.test_batch_size, shuffle=True, **kwargs)
#     else:
#         raise ValueError("No valid dataset is given.")
#     model.eval()
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             if args.cuda:
#                 data, target = data.cuda(), target.cuda()
#             data, target = Variable(data, volatile=True), Variable(target)
#             output = model(data)
#             pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
#             correct += pred.eq(target.data.view_as(pred)).cpu().sum()
#
#     print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
#         correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
#     return correct / float(len(test_loader.dataset))
#
#
# if __name__ == '__main__':
#     acc = test(model)
#
#     # Make real prune
#     print(cfg)
#     newmodel = vgg(dataset=args.dataset, cfg=cfg)
#     if args.cuda:
#         newmodel.cuda()
#
#     num_parameters = sum([param.nelement() for param in newmodel.parameters()])
#     savepath = os.path.join(args.save, "prune.txt")
#     with open(savepath, "w") as fp:
#         fp.write("Configuration: \n" + str(cfg) + "\n")
#         fp.write("Number of parameters: \n" + str(num_parameters) + "\n")
#         fp.write("Test accuracy: \n" + str(acc))
#
#     layer_id_in_cfg = 0
#     start_mask = torch.ones(3)  # 初始为三通道
#     end_mask = cfg_mask[layer_id_in_cfg]  # 第一层掩码，即下一层的输出，下下一层的输入
#     for [m0, m1] in zip(model.modules(), newmodel.modules()):
#         if isinstance(m0, nn.BatchNorm2d):
#             idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
#             if idx1.size == 1:
#                 idx1 = np.resize(idx1, (1,))
#             m1.weight.data = m0.weight.data[idx1.tolist()].clone()
#             m1.bias.data = m0.bias.data[idx1.tolist()].clone()
#             m1.running_mean = m0.running_mean[idx1.tolist()].clone()
#             m1.running_var = m0.running_var[idx1.tolist()].clone()
#             layer_id_in_cfg += 1
#             start_mask = end_mask.clone()
#             if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
#                 end_mask = cfg_mask[layer_id_in_cfg]
#         elif isinstance(m0, nn.Conv2d):
#             idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
#             idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
#             print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
#             if idx0.size == 1:
#                 idx0 = np.resize(idx0, (1,))
#             if idx1.size == 1:
#                 idx1 = np.resize(idx1, (1,))
#             w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
#             w1 = w1[idx1.tolist(), :, :, :].clone()
#             m1.weight.data = w1.clone()
#         elif isinstance(m0, nn.Linear):
#             idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
#             if idx0.size == 1:
#                 idx0 = np.resize(idx0, (1,))
#             m1.weight.data = m0.weight.data[:, idx0].clone()
#             m1.bias.data = m0.bias.data.clone()
#
#     torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))
#
#     print(newmodel)
#     model = newmodel
#     test(model)

# python vggprune.py --dataset cifar10 --depth 19 --percent 0.7 --model ./logs/model_best_vggnet_sr_93.78.pth.tar --save ./logs/vggprune
# python main.py --refine ./logs/pruned.pth.tar --dataset cifar10 --arch vgg --depth 19 --epochs 160 --save ./logs
