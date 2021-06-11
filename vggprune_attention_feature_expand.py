import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import *

'''
    vggprune_attention_feature.py: Prune VGG19 CIFAR10 Feature by attention-based method
    
    Baseline
    (1) Prune VGG19 with 70% proportion for cifar10 (Test accuracy: 32.61%)
    python vggprune_attention_feature.py --dataset cifar10 --depth 19 --percent 0.7 
        --model ./logs/baseline_vgg19_cifar10/model_best.pth.tar --save ./logs/attention_prune_feature_vgg19_percent_0.7
     
    (2) Prune VGG19 with 50% proportion for cifar10 (Test accuracy: 10.00%)
    python vggprune_attention_feature.py --dataset cifar10 --depth 19 --percent 0.5
        --model ./logs/baseline_vgg19_cifar10/model_best.pth.tar --save ./logs/attention_prune_feature_vgg19_percent_0.5
        
    
    Sparsity Cifar10
    (1) Prune Sparsity VGG19 with 70% proportion for cifar10 (Test accuracy: 27.91%%)
    python vggprune_attention_feature.py --dataset cifar10 --depth 19 --percent 0.7
        --model ./logs/sparsity_vgg19_cifar10_s_1e-4/model_best.pth.tar --save ./logs/attention_prune_feature_vgg19_sr_percent_0.7
    
    python vggprune_attention_feature.py --dataset cifar10 --depth 19 --percent 0.7
        --model logs/sparsity_vgg19_cifar10_s_1e_4/model_best.pth.tar --save logs/fine_tuning_vgg19_cifar10_feature_percent_0.7

    
    
    (2) Prune Sparsity VGG19 with 50% proportion for cifar10 (Test accuracy: 93.02%)
    python vggprune_attention_feature.py --dataset cifar10 --depth 19 --percent 0.5
        --model ./logs/sparsity_vgg19_cifar10_s_1e-4/model_best.pth.tar --save ./logs/attention_prune_feature_vgg19_sr_percent_0.5
    
    python vggprune_attention_feature.py --dataset cifar10 --depth 19 --percent 0.5
        --model ./logs/sparsity_vgg19_cifar10_s_1e_4/model_best.pth.tar 
        --save ./logs/prune_vgg19_cifar10_feature_percent_0.5
        
    Sparsity Cifar100    
    (3) Prune Sparsity VGG19 with 70% proportion for cifar100 (Test accuracy: 1.03%)
    python vggprune_attention_feature.py --dataset cifar100 --depth 19 --percent 0.7
        --model ./logs/attention_sparsity_vgg19_cifar100_s_1e-4/model_best.pth.tar --save ./logs/attention_sparsity_prune_feature_vgg19_cifar100_percent_0.7
    
    
    python vggprune_attention_feature.py --dataset cifar100 --depth 19 --percent 0.6
        --model ./logs/attention_sparsity_vgg19_cifar100_s_1e-4/model_best.pth.tar --save ./logs/attention_sparsity_prune_feature_vgg19_cifar100_percent_0.6

    python vggprune_attention_feature.py --dataset cifar100 --depth 19 --percent 0.5
        --model ./logs/attention_sparsity_vgg19_cifar100_s_1e-4/model_best.pth.tar --save ./logs/attention_sparsity_prune_feature_vgg19_cifar100_percent_0.5


    python vggprune_attention_feature_expand.py --dataset cifar100 --depth 19 --percent 0.6
        --model logs/sparsity_vgg19_cifar100_s_1e_4/model_best.pth.tar --save logs/prune_vgg19_cifar100_feature_expand_percent_0.6


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
parser.add_argument('--percent', type=float, default=0.7,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: None)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save pruned model (default: None)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

# Model
model = vgg(dataset=args.dataset, depth=args.depth)
if args.cuda:
    model.cuda()

best_prec1 = 0.
if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:.4f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.model))
    # print('Origin model: \r\n', model)


# Activation-based gamma
def activation_based_gamma(weight_data):
    d1, d2 = weight_data.shape[0], weight_data.shape[1]

    # 1. A: feature map data
    A = weight_data.view(d1, d2, -1).abs()
    c, h, w = A.shape

    # 2. Fsum(A): sum of values along the channel direction
    FsumA = torch.zeros(h, w).cuda()
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
        #         gamma[j] = (F_all - Fj).abs().sum()
        gamma[j] = torch.linalg.norm(F_all - Fj)

    return gamma


def get_dataloader():
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
        return test_loader
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
        return test_loader
    else:
        raise ValueError("No valid dataset is given.")


def test(model, test_loader):
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

    test_prec = float(correct) / len(test_loader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.4f})\n'
          .format(correct, len(test_loader.dataset), test_prec))
    return test_prec


if __name__ == '__main__':
    test_loader = get_dataloader()

    # one batch of Dataset
    idx, data_item = next(enumerate(test_loader))
    data1 = data_item[0][0].clone().unsqueeze(0)  # [1, 3, 32, 32]
    if args.cuda:
        data1 = data1.cuda()

    # Process
    # number of channels
    num_total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            num_total += m.weight.data.shape[0]
    gamma_list = torch.zeros(num_total)

    # all channels' gamma
    one_batch = data1.clone()
    index = 0
    for idx, m in enumerate(model.feature):
        one_batch = m(one_batch)
        if isinstance(m, nn.BatchNorm2d):
            value = one_batch.clone().squeeze(0)
            gamma = activation_based_gamma(value)
            size = value.shape[0]
            gamma_list[index:(index + size)] = gamma.clone()
            index += size

    # threshold
    y, i = torch.sort(gamma_list)
    thre_idx = int(num_total * args.percent)
    thre = y[thre_idx]

    # Pruned
    num_pruned, num_expand = 0, 0
    cfg = []
    cfg_mask = []
    one_batch = data1.clone()
    for k, m in enumerate(model.feature):
        one_batch = m(one_batch)
        if isinstance(m, nn.BatchNorm2d):
            value = one_batch.clone().squeeze(0)
            gammas = activation_based_gamma(value)
            mask = gammas.gt(thre).float().cuda()
            count_all = mask.shape[0]
            count_remain = int(torch.sum(mask))
            count_pruned = count_all - count_remain
            if count_pruned / count_all < 0.2:  # 裁剪率小于20%
                num = int(count_all * 1.2)
                cfg.append(num)
                num_expand += num - count_all
                print('layer index: {:d} \t total channel: {:d} \t expanding channel: {:d}'
                      .format(k, count_all, num))
            else:
                num = int(count_remain * 1.2)
                cfg.append(num)
                num_expand += int(count_remain * 0.2)
                num_pruned += count_pruned
                print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'
                      .format(k, count_all, num))

        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')

    pruned_ratio = num_pruned / num_total
    expand_ratio = num_expand / num_total
    all_ratio = pruned_ratio - expand_ratio
    print('Pre-processing Successful! Pruned ratio: {:4f} \t Expand ratio: {:4f} All ratio: {:4f}'
          .format(pruned_ratio, expand_ratio, all_ratio))

    # new model
    newmodel = vgg(dataset=args.dataset, cfg=cfg)
    if args.cuda:
        newmodel.cuda()
    print(cfg)

    acc = test(newmodel, test_loader)

    num_parameters = sum([param.nelement() for param in newmodel.parameters()])
    savepath = os.path.join(args.save, "prune.txt")
    with open(savepath, "w") as fp:
        fp.write("Configuration: \n{}\n".format(cfg))
        fp.write("Prune ratio: {:.4f}\n".format(pruned_ratio))
        fp.write("Number of parameters: {}\n".format(num_parameters))
        fp.write("Origin Model accuracy: {:.4f}\n".format(best_prec1))
        fp.write("Test Pruned Model accuracy: {:.4f}".format(acc))

    torch.save({'cfg': cfg}, os.path.join(args.save, 'pruned.pth.tar'))