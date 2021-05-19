from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import models


'''
    main.py: Train model with dataset & Save best model

    (1) Baseline: 
    VGG19 for cifar10 (Best accuracy: 0.9391)
    > python main.py --dataset cifar10 --arch vgg --depth 19 --save ./logs/baseline_vgg19_cifar10
    
    VGG19 for cifar100 (Best accuracy: 0.7255)
    > python main.py --dataset cifar100 --arch vgg --depth 19 --save ./logs/baseline_vgg19_cifar100
    
    ResNet for cifar10 (Best accuracy: 0.9507)
    > python main.py --dataset cifar10 --arch resnet --depth 164 --save ./logs/baseline_resnet164_cifar10

    (2) Sparsity: 
    BN:
    VGG19 for cifar10 & hyper-parameter sparsity 1e-4 (Best accuracy: 0.9347, 0.9366)
    > python main.py -sr --s 0.0001 --dataset cifar10 --arch vgg --depth 19 --save ./logs/sparsity_vgg19_cifar10_s_1e-4
    
    VGG19 for cifar100 & hyper-parameter sparsity 1e-4 (Best accuracy: 0.7269)
    > python main.py -sr --s 0.0001 --dataset cifar100 --arch vgg --depth 19 --save ./logs/sparsity_vgg19_cifar100_s_1e-4
    
    Attention Sparsity:
    VGG19 for cifar10 & hyper-parameter sparsity 1e-4 (Best accuracy: 0.9388)
    > python main_attention.py -sr --s 0.0001 --dataset cifar10 --arch vgg --depth 19 --save ./logs/attention_sparsity_vgg19_cifar10_s_1e-4
    
    VGG19 for cifar100 & hyper-parameter sparsity 1e-4 (Best accuracy: 72.49%)
    > python main_attention.py -sr --s 0.0001 --dataset cifar100 --arch vgg --depth 19 --save ./logs/attention_sparsity_vgg19_cifar100_s_1e-4
    
    ResNet for cifar10 (Best accuracy: 0.9480)
    > python main_attention.py --dataset cifar10 --arch resnet --depth 164 --save ./logs/attention_sparsity_resnet164_cifar10


    
    Activation-based
     VGG19 for cifar10 & hyper-parameter activation 1e-4 (Best accuracy: -)
    > python main_attention.py -sr --s 0.0001 --dataset cifar10 --arch vgg --depth 19 --save ./logs/activation_vgg19_cifar10_s_1e-4
    
    
    
    (3) Prune:
        VGG model references vggprune.py
        ResNet model references resprune.py
        DenseNet model references denseprune.py
    
    (4) Fine tune:
    
    
    Batch Normalization: VGG19 with 50% proportion for cifar10 (Best accuracy: 0.9373)
    > python main.py --refine ./logs/prune_vgg19_percent_0.5/pruned.pth.tar 
        --dataset cifar10 --arch vgg --depth 19 --epochs 160 --save ./logs/fine_tune_vgg19_percent_0.5
    
    Attention Weight: VGG19 with 50% proportion for cifar10 (Best accuracy: 0.9379)
     > python main.py --refine ./logs/attention_prune_vgg19_percent_0.5/pruned.pth.tar 
        --dataset cifar10 --arch vgg --depth 19 --epochs 160 --save ./logs/attention_fine_tune_vgg19_percent_0.5
        
    Attention Feature: VGG19 with 50% proportion for cifar10 (Best accuracy: 0.9387)
    > python main.py --refine ./logs/attention_prune_feature_vgg19_sr_percent_0.5/pruned.pth.tar 
        --dataset cifar10 --arch vgg --depth 19 --epochs 160 --save ./logs/attention_fine_tune_feature_vgg19_percent_0.5
        

    Batch Normalization: VGG19 with 70% proportion for cifar10 (Best accuracy: 0.9402)
    > python main.py --refine ./logs/prune_vgg19_percent_0.7/pruned.pth.tar 
        --dataset cifar10 --arch vgg --depth 19 --epochs 160 --save ./logs/fine_tune_vgg19_percent_0.7
    
    Batch Normalization: VGG19 with 50% proportion for cifar100 (Best accuracy: 0.7351)
    > python main.py --refine ./logs/prune_vgg19_cifar100_percent_0.5/pruned.pth.tar 
        --dataset cifar100 --arch vgg --depth 19 --epochs 160 --save ./logs/fine_tune_vgg19_cifar100_percent_0.5
    
        
    Attention Weight: VGG19 with 70% proportion for cifar10 (Best accuracy: 0.9259)
     > python main.py --refine ./logs/attention_prune_vgg19_percent_0.7/pruned.pth.tar 
        --dataset cifar10 --arch vgg --depth 19 --epochs 160 --save ./logs/attention_fine_tune_vgg19_percent_0.7
    
    
    Attention Feature: VGG19 with 50% proportion for cifar100 (Best accuracy: 0.7352)
    > python main.py --refine ./logs/attention_prune_feature_vgg19_sr_cifar100_percent_0.5/pruned.pth.tar 
        --dataset cifar100 --arch vgg --depth 19 --epochs 160 --save ./logs/attention_fine_tune_feature_vgg19_cifar100_percent_0.5
    
    
'''


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')  # Run sparsity regularization
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')         # Hyper-parameter sparsity (default 1e-4)
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='path to the pruned model to be fine tuned')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='  epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save prune model (default: none, current directory: ./ )')
parser.add_argument('--arch', default='vgg', type=str, 
                    help='architecture to use (vgg, resnet, densenet)')
parser.add_argument('--depth', default=19, type=int,
                    help='depth of the neural network')

# 0. Preset

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)


# 1. Dataset
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
else:
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

# 2. Model: fine-tune the pruned network
if args.refine:
    checkpoint = torch.load(args.refine)
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
    model.load_state_dict(checkpoint['state_dict'])
else:
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

if args.cuda:
    model.cuda()


# 3. Optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


# -1. Resume the process
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))


# Algorithm
def activation_based_gamma(weight_data):
    d1, d2 = weight_data.shape[0], weight_data.shape[1]

    # 1. A: feature map data
    A = weight_data.view(d1, d2, -1).abs()
    c, h, w = A.shape

    # 2. Fsum(A): sum of values along the channel direction
    FsumA = torch.zeros(h, w)
    for i in range(c):
        FsumA.add_(A[i])

    # 3. ||Fsum(A)||2: two norm
    FsumA_norm = torch.linalg.norm(FsumA)

    # 4. F(A) / ||F(A)||2: normalize weight data
    F_all = FsumA / FsumA_norm

    # 5. F(Aj) / ||F(Aj)||^2 & gamma = ∑ | F(A) / ||F(A)||2 - F(Aj) / ||F(Aj)||2 |
    gamma = torch.zeros(c)
    for j in range(c):
        FAj = FsumA - A[j]
        FAj_norm = torch.linalg.norm(FAj)
        Fj = FAj / FAj_norm
        # gamma[j] = (F_all - Fj).abs().sum()
        gamma[j] = torch.linalg.norm(F_all - Fj)

    return gamma



# additional subgradient descent on the sparsity-induced penalty term
# def updateBN():
#     for m in model.modules():
#         if isinstance(m, nn.BatchNorm2d):
#             m.weight.grad.data.add_(args.s * torch.sign(m.weight))  # 稀疏度惩罚项


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        if args.sr:  # sparsity regularization
            l1_loss = []
            for module in model.modules():
                if type(module) is nn.BatchNorm2d:
                    l1_loss.append(module.weight.abs().sum())
            loss += args.s * sum(l1_loss)

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))


def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))


if __name__ == '__main__':
    best_prec1 = 0.
    for epoch in range(args.start_epoch, args.epochs):
        if epoch in [args.epochs * 0.5, args.epochs * 0.75]:  # decent the learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        train(epoch)
        prec1 = test()
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, filepath=args.save)

    print("Best accuracy: " + str(best_prec1))