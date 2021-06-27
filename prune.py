import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from thop import profile
from thop import clever_format

import models
import utils
from models import channel_selection

'''
    vggprune.py: Prune vgg model with sparsity & Save to the local
        
    (1) Attention Transfer Pruning Method
    python vggprune.py --dataset cifar10 --depth 19 --percent 0.7 --pruning-method at
        --model logs/sparsity_vgg19_cifar10_s_1e_4/model_best.pth.tar --save logs/at_prune_vgg19_cifar10_percent_0.7
    
    python vggprune.py --dataset cifar100 --depth 19 --percent 0.5 --pruning-method at
        --model logs/sparsity_vgg19_cifar100_s_1e_4/model_best.pth.tar --save logs/at_prune_vgg19_cifar100_percent_0.5

    (2) Batch Normalization Pruning Method
    python vggprune.py --dataset cifar10 --depth 19 --percent 0.7 --pruning-method bn 
        --model logs/sparsity_vgg19_cifar10_s_1e_4/model_best.pth.tar --save logs/bn_prune_vgg19_cifar10_percent_0.7
    
    python prune.py --dataset cifar100 --depth 19 --percent 0.5 --pruning-method bn
        --model logs/sparsity_vgg19_cifar100_s_1e_4/model_best.pth.tar --save logs/bn_prune_vgg19_cifar100_percent_0.5_x
        
    
    resnet
    python prune.py --arch resnet --dataset cifar10 --depth 164 --percent 0.4 --pruning-method bn
        --model logs/sparsity_resnet164_cifar10_s_1e_4/model_best.pth.tar
        --save logs/bn_prune_resnet164_cifar10_percent_0.4
    
    python prune.py --arch resnet --dataset cifar10 --depth 164 --percent 0.4 --pruning-method bn
        --model logs/sparsity_resnet164_cifar10_s_1e_4/model_best.pth.tar 
        --save logs/bn_prune_resnet164_cifar10_percent_0.4
'''

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Network Slimming Pruning')
parser.add_argument('--arch', default='vgg', type=str,
                    help='architecture to use `vgg`, `resnet`, `densenet` (default: vgg)')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--num-thread', default=1, type=int, metavar="N",
                    help="number of dataloader working thread (default: 1)")
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training (default: False)')
parser.add_argument('--depth', type=int, default=19,
                    help='depth of the vgg model (default: 19)')
parser.add_argument('--percent', type=float, default=0.5,
                    help='model pruning rate (default: 0.5)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: None)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save pruned model (default: None)')
parser.add_argument('--pruning-method', default='bn', type=str, metavar='METHOD',
                    help='pruning method: `bn` means batch normalization; '
                         '`at` means attention transfer (default: bn, other value: at)')
parser.add_argument('--at-batch-size', type=int, default=32, metavar='N',
                    help='pruning method `at` is used for dataloader batch size,'
                         'but more than 32 will not have enough space')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)


def bn_threshold(model, percent):
    num_total = 0               # number of BatchNorm2d.weight
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            num_total += m.weight.data.shape[0]

    bn = torch.zeros(num_total)  # array of BatchNorm2d.weight
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index + size)] = m.weight.data.abs().clone()
            index += size

    y, i = torch.sort(bn)  # descending order (y: sort list, i: index list)
    thre_index = int(num_total * percent)   # threshold index
    thre = y[thre_index]                    # threshold value
    return thre, thre_index


def bn_prune_model(model, percent, cuda_available):
    threshold, thre_idx = bn_threshold(model, percent)
    if cuda_available:
        threshold = threshold.cuda()
    num_pruned, num_total = 0, 0
    cfg, cfg_mask = [], []
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(threshold).float()        # if value > threshold: True, else: False
            m.weight.data.mul_(mask)            # pruned weight
            m.bias.data.mul_(mask)              # pruned bias
            cfg.append(int(torch.sum(mask)))    # int: transfer tensor into numpy
            cfg_mask.append(mask.clone())
            num_total += mask.shape[0]
            num_pruned += mask.shape[0] - torch.sum(mask)  # num of pruned
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'
                  .format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')
    pruned_ratio = num_pruned / num_total
    print('Preprocess Successfully! Pruned ratio: {}'.format(pruned_ratio))
    print('Pruned cfg: {}'.format(cfg))
    return model, cfg, cfg_mask, pruned_ratio


def at_threshold(model, percent, one_batch):
    """
    :param model: model cuda
    :param percent: [0,1]
    :param one_batch: [batch_size, 3, 32, 32]
    :return threshold and threshold index
    """
    num_total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            num_total += m.weight.data.shape[0]
    gamma_list = torch.zeros(num_total)

    index = 0
    data = one_batch.clone()
    for idx, m in enumerate(model.feature):
        data = m(data)
        if isinstance(m, nn.BatchNorm2d):
            value = data.clone()
            gamma = utils.gammas(value)
            size = value.shape[1]
            gamma_list[index:(index + size)] = gamma.clone()
            index += size

    y, i = torch.sort(gamma_list)
    threshold_index = int(num_total * percent)
    threshold = y[threshold_index]
    return threshold, threshold_index


def at_prune_model(model, percent, one_batch, cuda_available):
    threshold, thre_idx = at_threshold(model, percent, one_batch)
    if cuda_available:
        threshold = threshold.cuda()
    num_pruned, num_total = 0, 0
    cfg, cfg_mask = [], []
    data = one_batch.clone()
    for k, m in enumerate(model.feature):
        data = m(data)
        if isinstance(m, nn.BatchNorm2d):
            value = data.clone().squeeze(0)
            gammas = utils.gammas(value)
            if cuda_available:
                gammas = gammas.cuda()
            mask = gammas.gt(threshold).float()
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            num_total += mask.shape[0]
            num_pruned += mask.shape[0] - torch.sum(mask)
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')
    pruned_ratio = num_pruned / num_total
    print('Preprocess Successfully! Pruned ratio: {}'.format(pruned_ratio))
    print('Pruned cfg: {}'.format(cfg))
    return model, cfg, cfg_mask, pruned_ratio


def generate_new_vgg_model(model, new_model, cfg_mask):
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)              # initial channel: 3
    end_mask = cfg_mask[layer_id_in_cfg]    #
    for [m0, m1] in zip(model.modules(), new_model.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()
    print('Generate new VGG Model successfully！')
    return new_model


def generate_new_resnet_model(model, new_model, cfg_mask):
    old_modules = list(model.modules())
    new_modules = list(new_model.modules())
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    conv_count = 0

    for layer_id in range(len(old_modules)):
        m0, m1 = old_modules[layer_id], new_modules[layer_id]
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            if isinstance(old_modules[layer_id + 1], channel_selection):
                # If the next layer is the channel selection layer,
                # then the current batchnorm 2d layer won't be pruned.
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                # We need to set the channel selection layer.
                m2 = new_modules[layer_id + 1]
                m2.indexes.data.zero_()
                m2.indexes.data[idx1.tolist()] = 1.0

                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
            else:
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            if conv_count == 0:
                m1.weight.data = m0.weight.data.clone()
                conv_count += 1
                continue
            if isinstance(old_modules[layer_id-1], channel_selection) or isinstance(old_modules[layer_id-1], nn.BatchNorm2d):
                # This convers the convolutions in the residual block.
                # The convolutions are either after the channel selection layer or after the batch normalization layer.
                conv_count += 1
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                # print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()

                # If the current convolution is not the last convolution in the residual block, then we can change the
                # number of output channels. Currently we use `conv_count` to detect whether it is such convolution.
                if conv_count % 3 != 1:
                    w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
                continue

            # We need to consider the case where there are downsampling convolutions.
            # For these convolutions, we just copy the weights.
            m1.weight.data = m0.weight.data.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))

            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()
    print('Generate new RseNet Model successfully！')
    return new_model


def save_model_record(save_path, cfg, origin_model, pruned_model, origin_dict, pruned_dict, cuda_available):
    input = torch.randn(1, 3, 32, 32)
    if cuda_available:
        input = input.cuda()
    flops1, params1 = profile(origin_model, inputs=(input, ))
    flops2, params2 = profile(pruned_model, inputs=(input, ))
    origin_dict['FLOPs Real'], origin_dict['Params Real'] = flops1, params1
    pruned_dict['FLOPs Real'], pruned_dict['Params Real'] = flops2, params2
    origin_dict['FLOPs'], origin_dict['Parameters'] = clever_format([flops1, params1], "%.2f")
    pruned_dict['FLOPs'], pruned_dict['Parameters'] = clever_format([flops2, params2], "%.2f")
    flops_ratio = float(flops1 - flops2) / flops1
    params_ratio = float(params1 - params2) / params1
    origin_dict['FLOPs Pruned'], origin_dict['Params Pruned'] = '-', '-'
    pruned_dict['FLOPs Pruned'] = '{:.2f}%'.format(flops_ratio * 100)
    pruned_dict['Params Pruned'] = '{:.2f}%'.format(params_ratio * 100)
    title_str, origin_str, pruned_str = '', '', ''
    for key in origin_dict.keys():
        title_str += key + ','
        origin_str += '{},'.format(origin_dict[key])
        pruned_str += '{},'.format(pruned_dict[key])
    with open(save_path, "w") as fp:
        fp.write("Configuration,{}\n".format(str(cfg).replace(',', '/')))
        fp.write(title_str + '\n')
        fp.write(origin_str + '\n')
        fp.write(pruned_str + '\n')


if __name__ == '__main__':
    # ==== dataset ====
    test_loader = utils.get_test_loader(args.dataset, args.test_batch_size, args.num_thread, args.cuda)

    # ==== model ====
    model, best_prec1 = utils.load_model(args.arch, args.dataset, args.depth, args.model, args.cuda)

    # ==== prune model ====
    if args.pruning_method == 'bn':
        pruned_model, cfg, cfg_mask, pruned_ratio = bn_prune_model(model, args.percent, args.cuda)
        acc_pruned, _ = utils.test(pruned_model, test_loader, args.cuda)
    elif args.pruning_method == 'at':
        data_loader = utils.get_test_loader(args.dataset, args.at_batch_size, args.num_thread, args.cuda)
        one_batch_data = utils.get_one_batch(data_loader, args.cuda)
        pruned_model, cfg, cfg_mask, pruned_ratio = at_prune_model(model, args.percent, one_batch_data, args.cuda)
        acc_pruned, _ = utils.test(pruned_model, test_loader, args.cuda)
    else:
        raise ValueError('Pruning Method does not exist.')

    # ==== new model ====
    new_model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=cfg)
    if args.cuda:
        new_model.cuda()
    if args.arch == 'vgg':
        new_model = generate_new_vgg_model(model, new_model, cfg_mask)
    elif args.arch == 'resnet':
        new_model = generate_new_resnet_model(model, new_model, cfg_mask)
    acc_new, _ = utils.test(new_model, test_loader, args.cuda)
    torch.save({'cfg': cfg, 'state_dict': new_model.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))

    # ==== save pruning record ====
    save_path = os.path.join(args.save, "pruning_record.csv")
    origin_data, pruned_data = {}, {}
    origin_data['Model'] = "vgg{}-{}".format(args.depth, args.dataset)
    pruned_data['Model'] = "{} ({:.0f}% {} Pruned)".format(
        origin_data['Model'], args.percent * 100, args.pruning_method.upper())
    origin_data['Test Error(%)'] = '{:.2f}'.format((1 - best_prec1) * 100)
    pruned_data['Test Error(%)'] = '{:.2f}'.format((1 - acc_pruned) * 100)
    origin_data['Accuracy'] = '{:.4f}'.format(best_prec1)
    if acc_pruned == acc_new:
        pruned_data['Accuracy'] = '{:.4f}'.format(acc_pruned)
    else:
        pruned_data['Accuracy'] = '{:.4f} (validate: {:.4f})'.format(acc_pruned, acc_new)
    save_model_record(save_path, cfg, model, new_model, origin_data, pruned_data, args.cuda)