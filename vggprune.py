import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from thop import profile
from thop import clever_format
import utils
from models import *


'''
    vggprune.py: Prune vgg model with sparsity & Save to the local
    
    (1) 70%
    Prune Sparsity VGG19 with 70% proportion for cifar10
    python vggprune.py --dataset cifar10 --depth 19 --percent 0.7 
        --model logs/sparsity_vgg19_cifar10_s_1e_4/model_best.pth.tar --save logs/prune_vgg19_cifar10_percent_0.7_2
    
    (2) 50%
    Prune Sparsity VGG19 with 50% proportion for cifar10 (Test accuracy: 93.47%)
    > python vggprune.py --dataset cifar10 --depth 19 --percent 0.5 
        --model ./logs/sparsity_vgg19_cifar10_s_1e-4/model_best.pth.tar --save ./logs/prune_vgg19_percent_0.5
    
    Prune Sparsity VGG19 with 50% proportion for cifar100 (Test accuracy: 1.36%)
    > python vggprune.py --dataset cifar100 --depth 19 --percent 0.5 
        --model ./logs/sparsity_vgg19_cifar100_s_1e-4/model_best.pth.tar --save ./logs/prune_vgg19_cifar100_percent_0.5
        
        
    (3) 30%
    Prune Sparsity VGG19 with 30% proportion for cifar10 (Test accuracy: 93.47%)
    python vggprune.py --dataset cifar10 --depth 19 --percent 0.3 
        --model ./logs/sparsity_vgg19_cifar10_s_1e-4/model_best.pth.tar --save ./logs/prune_vgg19_percent_0.3
        
        
    python vggprune.py --dataset cifar10 --depth 19 --percent 0.5
        --model ./logs/sparsity_vgg19_cifar10_s_1e_4/model_best.pth.tar --save ./logs/prune_vgg19_cifar10_percent_0.5_x

'''

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Network Slimming Pruning')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--num-thread', default=1, type=int, metavar="N",
                    help="number of dataloader working thread (default: 1)")
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=19,
                    help='depth of the vgg')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: None)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save pruned model (default: None)')
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


def bn_prune_model(model, percent):
    threshold, thre_idx = bn_threshold(model, percent)
    num_pruned, num_total = 0, 0
    cfg = []
    cfg_mask = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(threshold).float()        # if value > threshold: True, else: False
            num_total += mask.shape[0]
            num_pruned += mask.shape[0] - torch.sum(mask)   # num of pruned
            m.weight.data.mul_(mask)            # pruned weight
            m.bias.data.mul_(mask)              # pruned bias
            cfg.append(int(torch.sum(mask)))    # int: transfer tensor into numpy
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'
                  .format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')
    pruned_ratio = num_pruned / num_total
    print('Preprocess Successfully! Pruned ratio: ', pruned_ratio)
    print('Pruned cfg: {}'.format(cfg))
    return model, cfg, cfg_mask, pruned_ratio


def test(model, test_loader, cuda_available):
    model.eval()
    correct = 0
    for data, target in test_loader:
        if cuda_available:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_prec = float(correct) / len(test_loader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.4f})\n'
          .format(correct, len(test_loader.dataset), test_prec))
    return test_prec


def generate_new_model(model, new_model, cfg_mask):
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
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
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
    return new_model


def save_record(save_path, cfg, origin_model, pruned_model, origin_dict, pruned_dict):
    input = torch.randn(1, 3, 32, 32).cuda()
    flops1, params1 = profile(origin_model, inputs=(input, ))
    flops2, params2 = profile(pruned_model, inputs=(input, ))
    # flops2, params2 = 13422134, 2342342
    origin_dict['FLOPs'], origin_dict['Parameters'] = clever_format([flops1, params1], "%.2f")
    pruned_dict['FLOPs'], pruned_dict['Parameters'] = clever_format([flops2, params2], "%.2f")
    flops_ratio = float(flops1 - flops2) / flops1
    params_ratio = float(params1 - params2) / params1
    origin_dict['FLOPs Pruned'], origin_dict['Params Pruned'] = '-', '-'
    pruned_dict['FLOPs Pruned'] = '{:.4f}'.format(flops_ratio)
    pruned_dict['Params Pruned'] = '{:.4f}'.format(params_ratio)
    title_str, origin_str, pruned_str = '', '', ''
    for key in origin_dict.keys():
        title_str += key + ','
        origin_str += '{},'.format(origin_dict[key])
        pruned_str += '{},'.format(pruned_dict[key])

    with open(save_path, "w") as fp:
        fp.write("Configuration: \n{}\n".format(cfg))
        fp.write(title_str + '\n')
        fp.write(origin_str + '\n')
        fp.write(pruned_str + '\n')


if __name__ == '__main__':
    # ==== dataset ====
    test_loader = utils.get_test_loader(args.dataset, args.test_batch_size, args.num_thread, args.cuda)

    # ==== model ====
    model, best_prec1 = utils.generate_vgg_model(args.dataset, args.depth, args.model, args.cuda)

    # ==== prune model ====
    pruned_model, cfg, cfg_mask, pruned_ratio = bn_prune_model(model, args.percent)
    acc_pruned = test(pruned_model, test_loader, args.cuda)

    # ==== new model ====
    new_model = vgg(dataset=args.dataset, cfg=cfg)
    if args.cuda:
        new_model.cuda()
    new_model = generate_new_model(model, new_model, cfg_mask)
    acc_new = test(new_model, test_loader, args.cuda)
    torch.save({'cfg': cfg, 'state_dict': new_model.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))

    save_path = os.path.join(args.save, "pruning_record.csv")
    origin_data, pruned_data = {}, {}
    origin_data['Model'] = "vgg{}-{}".format(args.depth, args.dataset)
    pruned_data['Model'] = origin_data['Model'] + "({}% Pruned)".format(args.percent * 100)
    origin_data['Accuracy'] = best_prec1
    pruned_data['Accuracy'] = '{:.4f} (validate: {:.4f})'.format(acc_pruned, acc_new)
    save_record(save_path, cfg, model, new_model, origin_data, pruned_data)
