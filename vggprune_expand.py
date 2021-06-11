import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import *


'''
    vggprune.py: Prune vgg model with sparsity & Save to the local
    
    (1) 70%
    Prune Sparsity VGG19 with 70% proportion for cifar10 (Test accuracy: 19.17%)
    python vggprune_expand.py --dataset cifar10 --depth 19 --percent 0.7 
        --model ./logs/sparsity_vgg19_cifar10_s_1e_4/model_best.pth.tar --save ./logs/prune_vgg19_cifar10_expand_percent_0.7
    
    0.7321 (origin, not expand the pruned layer)
    python vggprune_expand.py --dataset cifar100 --depth 19 --percent 0.5
        --model ./logs/sparsity_vgg19_cifar100_s_1e-4/model_best.pth.tar --save ./logs/prune_expand_vgg19_cifar100_percent_0.5
    
    
    python vggprune_expand.py --dataset cifar100 --depth 19 --percent 0.5
        --model ./logs/sparsity_vgg19_cifar100_s_1e-4/model_best.pth.tar --save ./logs/prune_expand_more_vgg19_cifar100_percent_0.5
    
    python vggprune_expand.py --dataset cifar100 --depth 19 --percent 0.6
        --model logs/sparsity_vgg19_cifar100_s_1e_4/model_best.pth.tar 
        --save logs/prune_vgg19_cifar100_expand_percent_0.6
    

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

# Number of BatchNorm2d.weight
total = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        total += m.weight.data.shape[0]

# Array of BatchNorm2d.weight
bn = torch.zeros(total)
index = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn[index:(index+size)] = m.weight.data.abs().clone()
        index += size

# Threshold
y, i = torch.sort(bn)   # descending order, y: sort list, i: index list
thre_index = int(total * args.percent)  # threshold index
thre = y[thre_index]  # threshold value


num_pruned, num_expand = 0, 0
num_cfg = []
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        weight_copy = m.weight.data.abs().clone()
        mask = weight_copy.gt(thre).float().cuda()
        count_all = mask.shape[0]
        count_remain = int(torch.sum(mask))
        count_pruned = count_all - count_remain
        if count_pruned / count_all < 0.2:  # 裁剪率小于20%
            num = int(count_all * 1.2)
            num_cfg.append(num)
            num_expand += num - count_all
            print('layer index: {:d} \t total channel: {:d} \t expanding channel: {:d}'
                  .format(k, count_all, num))
        else:
            num = int(count_remain * 1.2)
            num_cfg.append(num)
            num_expand += int(count_remain * 0.2)
            num_pruned += count_pruned
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'
                  .format(k, count_all, count_pruned))
    elif isinstance(m, nn.MaxPool2d):
        num_cfg.append('M')

pruned_ratio = num_pruned / total
expand_ratio = num_expand / total
all_ratio = pruned_ratio - expand_ratio

print('Pre-processing Successful! Pruned ratio: {:4f} \t Expand ratio: {:4f} All ratio: {:4f}'
      .format(pruned_ratio, expand_ratio, all_ratio))


# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    correct = 0

    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))


if __name__ == '__main__':
    acc = test(model)

    # Make real prune
    print(num_cfg)
    newmodel = vgg(dataset=args.dataset, cfg=num_cfg)
    if args.cuda:
        newmodel.cuda()

    num_parameters = sum([param.nelement() for param in newmodel.parameters()])
    savepath = os.path.join(args.save, "prune.txt")
    with open(savepath, "w") as fp:
        fp.write("Configuration: \n{}\n".format(num_cfg))
        fp.write("Prune ratio: {:.4f}\n".format(pruned_ratio))
        fp.write("Expand ratio: {:.4f}\n".format(expand_ratio))
        fp.write("Number of parameters: {}\n".format(num_parameters))
        fp.write("Test Pruned Model accuracy: {:.4f}".format(acc))

    torch.save({'cfg': num_cfg}, os.path.join(args.save, 'pruned.pth.tar'))

    print(newmodel)
    model = newmodel
    test(model)
