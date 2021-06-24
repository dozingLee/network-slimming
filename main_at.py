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
import utils

'''
    python main_attention_transfer.py 
        --refine logs/prune_vgg19_cifar100_percent_0.5/pruned.pth.tar --init-weight
        --large logs/sparsity_vgg19_cifar100_s_1e_4/model_best.pth.tar
        --dataset cifar100 --arch vgg --depth 19 --epochs 160 
        --save logs/attention_transfer_vgg19_cifar100_2_4_8_12_beta_1
        --conv-cfg 2 4 8 12 --beta 1
        
    
    python main_attention_transfer.py 
        --refine logs/prune_vgg19_cifar100_percent_0.5/pruned.pth.tar
        --large logs/sparsity_vgg19_cifar100_s_1e_4/model_best.pth.tar
        --dataset cifar100 --arch vgg --depth 19 --epochs 160 
        --save logs/attention_transfer_vgg19_cifar100_2_4_8_12_beta_100_alpha_0.9
        --conv-cfg 2 4 8 12 --beta 100 --alpha 0.9
        
        
    python main_attention_transfer.py 
        --refine logs/prune_vgg19_cifar10_percent_0.7/pruned.pth.tar
        --large logs/sparsity_vgg19_cifar10_s_1e_4/model_best.pth.tar
        --dataset cifar10 --arch vgg --depth 19 --epochs 160 
        --save logs/attention_transfer_vgg19_cifar10_2_4_8_12_beta_100_alpha_0.9
        --conv-cfg 2 4 8 12 --beta 100 --alpha 0.9
        
    
    python main_attention_transfer.py 
        --refine logs/prune_feature_vgg19_cifar100_percent_0.5/pruned.pth.tar
        --large logs/sparsity_vgg19_cifar100_s_1e_4/model_best.pth.tar
        --dataset cifar100 --arch vgg --depth 19 --epochs 160 
        --save logs/attention_transfer_vgg19_cifar100_2_4_8_12_beta_100_alpha_0.9_2
        --conv-cfg 2 4 8 12 --beta 100 --alpha 0.9
    
    python main_attention_transfer.py 
        --refine logs/prune_feature_vgg19_cifar100_percent_0.5/pruned.pth.tar
        --large logs/sparsity_vgg19_cifar100_s_1e_4/model_best.pth.tar
        --dataset cifar100 --arch vgg --depth 19 --epochs 160 
        --save logs/attention_transfer_vgg19_cifar100_2_4_8_12_beta_1000_alpha_0.9_2
        --conv-cfg 2 4 8 12 --beta 1000 --alpha 0.9
'''

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')  # Run sparsity regularization
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')  # Hyper-parameter sparsity (default 1e-4)
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='start epoch number (useful on restarts)')
parser.add_argument('--nthread', default=1, type=int, metavar="N",
                    help="number of dataloader working thread (default: 1)")
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
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='path to the pruned model to be fine tuned')
parser.add_argument('--large', default='', type=str, metavar='PATH',
                    help='path to the large model to be refined')
parser.add_argument('--conv-cfg', default=[], type=int, nargs='*',
                    help='refine convolution channel index (start with 1)')
parser.add_argument('--alpha', default=0, type=float, help='hyper-parameter for AT+KD')
parser.add_argument('--beta', default=0, type=float, help='hyper-parameter for AT')
parser.add_argument('--temperature', default=4, type=float, metavar="T",
                    help='Knowledge Distillation temperature')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
utils.init_seeds(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)


def generate_models(args):
    """
    Student Model: pruned model
        file required: pruned strategy ['cfg']

    Teacher Model: origin model
        file required: model parameter ['state_dict']

    Common attributes:
        dataset, depth
        conv_cfg: return convolution values
    """
    if not args.refine:
        raise ValueError('Parameter `refine` value is empty!')
    file_s = torch.load(args.refine)
    model_s = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=file_s['cfg'],
                                         conv_cfg=args.conv_cfg)
    if not args.init_weight and 'state_dict' in file_s:
        model_s.load_state_dict(file_s['state_dict'])

    file_t = torch.load(args.large)
    model_t = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, conv_cfg=args.conv_cfg)
    if 'state_dict' not in file_t:
        raise ValueError('Teacher Model parameters are empty!')
    model_t.load_state_dict(file_t['state_dict'])

    if args.cuda:
        model_s.cuda(), model_t.cuda()

    return model_s, model_t, file_s['cfg']


def train(model_s, model_t, epoch, data_loader):
    model_s.train(), model_t.eval()
    for batch_idx, (data, target) in enumerate(data_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        with torch.no_grad():
            y_t, g_t = model_t(data)
        y_s, g_s = model_s(data)

        loss_ori = utils.distillation(y_s, y_t, target, args.temperature, args.alpha)
        loss_groups = [utils.at_loss(x, y).sum() for x, y in zip(g_t, g_s)]
        loss = loss_ori + args.beta * sum(loss_groups)
        loss.backward()

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                       100. * batch_idx / len(data_loader), loss.item()))


def test(model, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in data_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            output, output_s = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    # (The divisor is float and the dividend is long, but the result is long)
    test_loss /= len(data_loader.dataset)  # loss mean
    test_prec = float(correct) / len(data_loader.dataset)  # prec
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f})\n'.format(
        test_loss, correct, len(data_loader.dataset), test_prec))
    return test_prec, test_loss


if __name__ == '__main__':
    train_loader, test_loader = utils.get_dataset_loaders(
        args.dataset, args.batch_size, args.test_batch_size, args.nthread, args.cuda)

    model_s, model_t, cfg = generate_models(args)

    optimizer = optim.SGD(model_s.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    record_file = os.path.join(args.save, 'record.csv')
    with open(record_file, 'w+') as f:
        f.write('epoch,loss,prec\n')

    # ====== Training ======
    best_prec = 0.
    for epoch in range(args.epochs):
        if epoch in [args.epochs * 0.5, args.epochs * 0.75]:  # decent the learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        train(model_s, model_t, epoch, train_loader)

        prec1, loss1 = test(model_s, test_loader)  # process test
        is_best = prec1 > best_prec  # save the best
        best_prec = max(prec1, best_prec)

        save_cfg = {
            'epoch': epoch,
            'state_dict': model_s.state_dict(),
            'best_prec': best_prec,
            'cfg': cfg,
            'optimizer': optimizer.state_dict(),
            'alpha': args.alpha,
            'beta': args.beta,
            'temperature': args.temperature,
            'conv_cfg': args.conv_cfg
        }
        utils.save_checkpoint(save_cfg, is_best, args.save)

        with open(record_file, 'a+') as f:
            f.write('{},{:.4f},{:.4f}\n'.format(epoch, loss1, prec1))

    utils. visualization_record(args.save)
    print("Best accuracy: {:.4f}".format(best_prec))
