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
import utils_prune
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
    
    python prune.py --dataset cifar100 --depth 19 --percent 0.5 --pruning-method at
        --model logs/sparsity_vgg19_cifar100_s_1e_4/model_best.pth.tar --save logs/at_prune_vgg19_cifar100_percent_0.5_x_1
        
    
    resnet
    python prune.py --arch resnet --dataset cifar10 --depth 164 --percent 0.4 --pruning-method bn
        --model logs/sparsity_resnet164_cifar10_s_1e_4/model_best.pth.tar
        --save logs/bn_prune_resnet164_cifar10_percent_0.4
    
    python prune.py --arch resnet --dataset cifar10 --depth 164 --percent 0.4 --pruning-method bn
        --model logs/sparsity_resnet164_cifar10_s_1e_4/model_best.pth.tar 
        --save logs/bn_prune_resnet164_cifar10_percent_0.4
        
    python prune.py --arch resnet --dataset cifar10 --depth 164 --percent 0.4 --pruning-method at
        --model logs/sparsity_resnet164_cifar10_s_1e_4/model_best.pth.tar 
        --save logs/at_prune_resnet164_cifar10_percent_0.4
    
    python prune.py --arch resnet --dataset cifar100 --depth 164 --percent 0.4 --pruning-method bn
        --model logs/sparsity_resnet164_cifar100_s_1e_4/model_best.pth.tar 
        --save logs/bn_prune_resnet164_cifar100_percent_0.4
    
    python prune.py --arch resnet --dataset cifar100 --depth 164 --percent 0.4 --pruning-method bn
        --model logs/sparsity_resnet164_cifar100_s_1e_4/model_best.pth.tar 
        --save logs/bn_prune_resnet164_cifar100_percent_0.4
    
    python prune.py --arch resnet --dataset cifar10 --depth 164 --percent 0.6 --pruning-method bn
        --model logs/sparsity_resnet164_cifar10_s_1e_4/model_best.pth.tar 
        --save logs/bn_prune_resnet164_cifar10_percent_0.6
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
        fp.write(title_str + '\n')
        fp.write(origin_str + '\n')
        fp.write(pruned_str + '\n')
        fp.write("Configuration,{}\n".format(str(cfg).replace(',', '/')))


if __name__ == '__main__':
    # ==== dataset ====
    test_loader = utils.get_test_loader(args.dataset, args.test_batch_size, args.num_thread, args.cuda)

    # ==== model ====
    model, best_prec1 = utils.load_model(args.arch, args.dataset, args.depth, args.model, args.cuda)

    # ==== prune model ====
    if args.pruning_method == 'bn':
        pruned_model, cfg, cfg_mask, pruned_ratio = utils_prune.bn_prune_model(model, args.percent, args.cuda)
        acc_pruned, _ = utils.test(pruned_model, test_loader, args.cuda)
    elif args.pruning_method == 'at':
        data_loader = utils.get_test_loader(args.dataset, args.at_batch_size, args.num_thread, args.cuda)
        one_batch_data = utils.get_one_batch(data_loader, args.cuda)
        if args.arch == 'vgg':
            pruned_model, cfg, cfg_mask, pruned_ratio = utils_prune.at_vgg_prune_model(
                    model, args.percent, one_batch_data, args.cuda)
        elif args.arch == 'resnet':
            pruned_model, cfg, cfg_mask, pruned_ratio = utils_prune.at_resnet_prune_model(
                    model, args.percent, one_batch_data, args.cuda)
        else:
            raise ValueError("`arch` is not found.")
        acc_pruned, _ = utils.test(pruned_model, test_loader, args.cuda)
    else:
        raise ValueError('Pruning Method does not exist.')

    # ==== new model ====
    new_model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=cfg)
    if args.cuda:
        new_model.cuda()
    if args.arch == 'vgg':
        new_model = utils_prune.generate_new_vgg_model(model, new_model, cfg_mask)
    elif args.arch == 'resnet':
        new_model = utils_prune.generate_new_resnet_model(model, new_model, cfg_mask)
    elif args.arch == 'densenet':
        new_model = utils_prune.generate_new_densenet_model(model, new_model, cfg_mask)
    else:
        raise ValueError('Generate new model failed.')
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