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
import pandas as pd
import matplotlib.pyplot as plt


'''
    main.py: Train model with dataset & Save best model

    (1) Baseline: 
    VGG19 cifar10 (Best accuracy: 0.9391)
    python main.py --dataset cifar10 --arch vgg --depth 19 --save ./logs/baseline_vgg19_cifar10_2 --init-weight
    
    VGG19 cifar100 (Best accuracy: 0.7255)
    > python main.py --dataset cifar100 --arch vgg --depth 19 --save ./logs/baseline_vgg19_cifar100 --init-weight
    
    ResNet cifar10 (Best accuracy: 0.9507)
    > python main.py --dataset cifar10 --arch resnet --depth 164 --save ./logs/baseline_resnet164_cifar10


    (2) Sparsity: 
    VGG19 for cifar10 & hyper-parameter sparsity 1e-4 (Best accuracy: 0.9347)
    > python main.py -sr --s 0.0001 --dataset cifar10 --arch vgg --depth 19 --save ./logs/sparsity_vgg19_cifar10_s_1e_4
    
    VGG19 for cifar100 & hyper-parameter sparsity 1e-4 (Best accuracy: 0.7269)
    > python main.py -sr --s 0.0001 --dataset cifar100 --arch vgg --depth 19 --save ./logs/sparsity_vgg19_cifar100_s_1e_4
    
    
    0.9393
    python main.py -sr --s 0.0001 --dataset cifar10 --arch vgg --depth 19 --save ./logs/sparsity_vgg19_cifar10_s_1e_4
    
    0.9351
    python main.py -sr --s 0.00001 --dataset cifar10 --arch vgg --depth 19 --save ./logs/sparsity_vgg19_cifar10_s_1e_5
    
    0.7222
    python main.py -sr --s 0.0001 --dataset cifar100 --arch vgg --depth 19 --save ./logs/sparsity_vgg19_cifar100_s_1e_4
    
    


    
    (3) Prune:
        VGG model references vggprune.py
        ResNet model references resprune.py
        DenseNet model references denseprune.py
    
    
    (4) Fine tune:
    
    Batch Normalization:
    
    (1) Cifar10 VGG19 with 50% proportion (Best accuracy: 0.9373)
    python main.py --refine ./logs/prune_vgg19_percent_0.5/pruned.pth.tar 
        --dataset cifar10 --arch vgg --depth 19 --epochs 160 --save ./logs/fine_tune_vgg19_percent_0.5
    
    (2) Cifar10 VGG19 with 70% proportion (Best accuracy: 0.9402)
    python main.py --refine ./logs/prune_vgg19_percent_0.7/pruned.pth.tar 
        --dataset cifar10 --arch vgg --depth 19 --epochs 160 --save ./logs/fine_tune_vgg19_percent_0.7
        
    (3) Cifar100 VGG19 with 50% proportion for cifar100 (Best accuracy: 0.7351)
    python main.py --refine ./logs/prune_vgg19_cifar100_percent_0.5/pruned.pth.tar 
        --dataset cifar100 --arch vgg --depth 19 --epochs 160 --save ./logs/fine_tune_vgg19_cifar100_percent_0.5
    
    Attention Weight:
    
    (1) Cifar10 VGG19 with 50% proportion (Best accuracy: 0.9379)
    python main.py --refine ./logs/attention_prune_vgg19_percent_0.5/pruned.pth.tar 
        --dataset cifar10 --arch vgg --depth 19 --epochs 160 --save ./logs/attention_fine_tune_vgg19_percent_0.5
    
    (2) Cifar10 VGG19 with 70% proportion (Best accuracy: 0.9259)
     > python main.py --refine ./logs/attention_prune_vgg19_percent_0.7/pruned.pth.tar 
        --dataset cifar10 --arch vgg --depth 19 --epochs 160 --save ./logs/attention_fine_tune_vgg19_percent_0.7
    
    Attention Feature:
    
    (1) Cifar10 VGG19 with 50% proportion (Best accuracy: 0.9387)
    python main.py --refine ./logs/attention_prune_feature_vgg19_sr_percent_0.5/pruned.pth.tar 
        --dataset cifar10 --arch vgg --depth 19 --epochs 160 --save ./logs/attention_fine_tune_feature_vgg19_percent_0.5
    
    (2) Cifar10 VGG19 with 70% proportion (Best accuracy: 0.9363)
    python main.py --refine ./logs/attention_prune_feature_vgg19_sr_percent_0.7/pruned.pth.tar 
        --dataset cifar10 --arch vgg --depth 19 --epochs 160 --save ./logs/attention_fine_tune_feature_vgg19_percent_0.7
    
    (3) Cifar100 VGG19 with 50% proportion (Best accuracy: 0.7352)
    > python main.py --refine ./logs/attention_prune_feature_vgg19_sr_cifar100_percent_0.5/pruned.pth.tar 
        --dataset cifar100 --arch vgg --depth 19 --epochs 160 --save ./logs/attention_fine_tune_feature_vgg19_cifar100_percent_0.5
     
    Attention Sparsity:
    
    (1) Sparsity Cifar10 VGG19 with 70% proportion (Best Accuracy: 0.9388)
    python main.py --refine ./logs/attention_sparsity_prune_feature_vgg19_percent_0.7/pruned.pth.tar 
        --dataset cifar10 --arch vgg --depth 19 --epochs 160 --save ./logs/attention_sparsity_fine_tune_feature_vgg19_percent_0.7
    
    (2) Sparstiy Cifar100 VGG19 with 70% proportion (Best Accuracy: 70.20%)
    python main.py --refine ./logs/attention_sparsity_prune_feature_vgg19_cifar100_percent_0.7/pruned.pth.tar 
        --dataset cifar100 --arch vgg --depth 19 --epochs 160 --save ./logs/attention_sparsity_fine_tune_feature_vgg19_cifar100_percent_0.7
    
    (3) Sparstiy Cifar100 VGG19 with 60% proportion (Best Accuracy: 72.59%)
    python main.py --refine ./logs/attention_sparsity_prune_feature_vgg19_cifar100_percent_0.6/pruned.pth.tar 
        --dataset cifar100 --arch vgg --depth 19 --epochs 160 --save ./logs/attention_sparsity_fine_tune_feature_vgg19_cifar100_percent_0.6
    
    (3) Sparstiy Cifar100 VGG19 with 50% proportion (Best Accuracy: 73.23%)
    python main.py --refine ./logs/attention_sparsity_prune_feature_vgg19_cifar100_percent_0.5/pruned.pth.tar 
        --dataset cifar100 --arch vgg --depth 19 --epochs 160 --save ./logs/attention_sparsity_fine_tune_feature_vgg19_cifar100_percent_0.5
    
    
    Random Weight:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    (1) VGG19 Cifar10 with 50% proportion (Best Accuracy: 0.9362)
    python main.py --refine ./logs/prune_vgg19_percent_0.5/pruned.pth.tar --init-weight
        --dataset cifar10 --arch vgg --depth 19 --epochs 160 --save ./logs/random_fine_tune_vgg19_percent_0.5
        
        
   #  python main.py --refine ./logs/prune_expand_vgg19_percent_0.7/pruned.pth.tar --init-weight
   #      --dataset cifar10 --arch vgg --depth 19 --epochs 160 --save ./logs/fine_tune_expand_vgg19_percent_0.7
   #  
   #  
   #  python main.py --refine ./logs/prune_expand_vgg19_cifar100_percent_0.5/pruned.pth.tar --init-weight
   #      --dataset cifar100 --arch vgg --depth 19 --epochs 160 --save ./logs/fine_tune_expand_vgg19_cifar100_percent_0.5
   #      
   #  
   # python main.py --refine ./logs/prune_expand_more_vgg19_cifar100_percent_0.5/pruned.pth.tar --init-weight
   #      --dataset cifar100 --arch vgg --depth 19 --epochs 160 --save ./logs/fine_tune_expand_more_vgg19_cifar100_percent_0.5
        
    
    python main.py --refine logs/prune_vgg19_cifar100_percent_0.5/pruned.pth.tar
        --dataset cifar100 --arch vgg --depth 19 --epochs 160 --save logs/fine_tuning_vgg19_cifar100_percent_0.5
        
    python main.py --refine logs/prune_vgg19_cifar100_percent_0.5/pruned.pth.tar --init-weight
        --dataset cifar100 --arch vgg --depth 19 --epochs 160 --save logs/fine_tuning_vgg19_cifar100_percent_0.5_init

    0.7309
    python main.py --refine logs/prune_vgg19_cifar100_percent_0.5/pruned.pth.tar --lr 0.05
        --dataset cifar100 --arch vgg --depth 19 --epochs 160 --save logs/fine_tuning_vgg19_cifar100_percent_0.5_lr_0.05
    
    python main.py --refine logs/prune_vgg19_cifar100_feature_percent_0.5/pruned.pth.tar
        --dataset cifar100 --arch vgg --depth 19 --epochs 160 --save logs/fine_tuning_vgg19_cifar100_feature_percent_0.5
    
    
    python main.py --refine logs/prune_vgg19_cifar10_expand_percent_0.7/pruned.pth.tar --init-weight
        --dataset cifar10 --arch vgg --depth 19 --epochs 160 --save logs/fine_tuning_vgg19_cifar10_expand_percent_0.7
        
    python main.py --refine logs/fine_tuning_vgg19_cifar10_feature_percent_0.7/pruned.pth.tar 
        --dataset cifar10 --arch vgg --depth 19 --epochs 160 --save logs/fine_tuning_vgg19_cifar10_feature_percent_0.7
    
    python main.py --refine logs/prune_vgg19_cifar10_feature_percent_0.5/pruned.pth.tar 
        --dataset cifar10 --arch vgg --depth 19 --epochs 160 --save logs/fine_tuning_vgg19_cifar10_feature_percent_0.5
    
    python main.py --refine logs/prune_vgg19_cifar10_percent_0.5/pruned.pth.tar 
        --dataset cifar10 --arch vgg --depth 19 --epochs 160 --save logs/fine_tuning_vgg19_cifar10_percent_0.5
    
    python main.py --refine logs/prune_vgg19_cifar100_feature_expand_percent_0.6/pruned.pth.tar --init-weight
        --dataset cifar100 --arch vgg --depth 19 --epochs 160 --save logs/prune_vgg19_cifar100_feature_expand_percent_0.6
        
    
    python main.py --refine logs/fine_tuning_vgg19_cifar10_feature_3_percent_0.7/pruned.pth.tar --lr 0.15
        --dataset cifar10 --arch vgg --depth 19 --epochs 160 --save logs/fine_tuning_vgg19_cifar10_feature_4_percent_0.7
    
    python main.py --refine logs/prune_feature_vgg19_cifar100_percent_0.5/pruned.pth.tar
        --dataset cifar100 --arch vgg --depth 19 --epochs 160 --save logs/prune_feature_vgg19_cifar100_percent_0.5
    
    
'''

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')  # Run sparsity regularization
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')  # Hyper-parameter sparsity (default 1e-4)
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
parser.add_argument('--init-weight', action='store_true', default=False,
                    help='initialize model weight')

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
model_cfg = []
if args.refine:
    checkpoint = torch.load(args.refine)
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    model_cfg = checkpoint['cfg']
else:
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

if args.init_weight:
    model._initialize_weights()

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


# additional subgradient descent on the sparsity-induced penalty term
def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s * torch.sign(m.weight.data))  # 稀疏度惩罚项


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target, reduction='mean')
        loss.backward()
        if args.sr:  # sparsity regularization
            updateBN()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
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
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    # (The divisor is float and the dividend is long, but the result is long)
    test_loss /= len(test_loader.dataset)               # loss mean
    test_prec = float(correct) / len(test_loader.dataset)     # prec
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f})\n'.format(
        test_loss, correct, len(test_loader.dataset), test_prec))
    return test_prec, test_loss


def save_checkpoint(state, is_best, save_path):
    torch.save(state, os.path.join(save_path, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(save_path, 'checkpoint.pth.tar'), os.path.join(save_path, 'model_best.pth.tar'))


def visualization_record(best_prec, save_path):
    data = pd.read_csv(os.path.join(save_path, 'record.csv'))
    line_loss, = plt.plot(data['loss'], 'r-')
    line_prec, = plt.plot(data['prec'], 'b-')
    plt.legend([line_loss, line_prec], ['loss', 'accuracy'], loc='upper right')
    plt.ylabel('value', fontsize=12)
    plt.xlabel('epoch', fontsize=12)
    plt.title('Train loss and accuracy (best_prec: {})'.format(best_prec), fontsize=14)
    plt.savefig(os.path.join(save_path, "record train loss.png"))
    print('Save the training loss and accuracy successfully.')


if __name__ == '__main__':
    # clean record file
    record_file = os.path.join(args.save, 'record.csv')
    if os.path.exists(record_file):
        os.remove(record_file)
    with open(record_file, 'w+') as f:
        f.write('epoch,loss,prec\n')

    # ====== Training ======
    best_prec1 = 0.
    for epoch in range(args.start_epoch, args.epochs):
        if epoch in [args.epochs * 0.5, args.epochs * 0.75]:  # decent the learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        train(epoch)                    # process train
        prec1, loss1 = test()           # process test
        is_best = prec1 > best_prec1    # save the best
        best_prec1 = max(prec1, best_prec1)

        if model_cfg:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'cfg': model_cfg,
                'optimizer': optimizer.state_dict()
            }, is_best, save_path=args.save)
        else:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict()
            }, is_best, save_path=args.save)

        with open(record_file, 'a+') as f:
            f.write('{},{:.4f},{:.4f}\n'.format(epoch, loss1, prec1))

    visualization_record(best_prec1, args.save)
    print("Best accuracy: {:.4f}".format(best_prec1))
