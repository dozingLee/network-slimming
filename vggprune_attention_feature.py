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
    --model ./logs/attention_sparsity_vgg19_cifar10_s_1e-4/model_best.pth.tar --save ./logs/attention_sparsity_prune_feature_vgg19_percent_0.7

    
    
    (2) Prune Sparsity VGG19 with 50% proportion for cifar10 (Test accuracy: 93.02%)
    python vggprune_attention_feature.py --dataset cifar10 --depth 19 --percent 0.5
        --model ./logs/sparsity_vgg19_cifar10_s_1e-4/model_best.pth.tar --save ./logs/attention_prune_feature_vgg19_sr_percent_0.5
        
        
    Sparsity Cifar100    
    (3) Prune Sparsity VGG19 with 70% proportion for cifar100 (Test accuracy: 1.03%)
    python vggprune_attention_feature.py --dataset cifar100 --depth 19 --percent 0.7
        --model ./logs/attention_sparsity_vgg19_cifar100_s_1e-4/model_best.pth.tar --save ./logs/attention_sparsity_prune_feature_vgg19_cifar100_percent_0.7
    
    
    python vggprune_attention_feature.py --dataset cifar100 --depth 19 --percent 0.6
        --model ./logs/attention_sparsity_vgg19_cifar100_s_1e-4/model_best.pth.tar --save ./logs/attention_sparsity_prune_feature_vgg19_cifar100_percent_0.6

    python vggprune_attention_feature.py --dataset cifar100 --depth 19 --percent 0.5
        --model ./logs/attention_sparsity_vgg19_cifar100_s_1e-4/model_best.pth.tar --save ./logs/attention_sparsity_prune_feature_vgg19_cifar100_percent_0.5


    python vggprune_attention_feature.py --dataset cifar100 --depth 19 --percent 0.5
        --model logs/sparsity_vgg19_cifar100_s_1e_4/model_best.pth.tar --save logs/prune_feature_vgg19_cifar100_percent_0.5


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
    num_pruned = 0
    cfg = []
    cfg_mask = []
    one_batch = data1.clone()
    for k, m in enumerate(model.feature):
        one_batch = m(one_batch)
        if isinstance(m, nn.BatchNorm2d):
            value = one_batch.clone().squeeze(0)
            gammas = activation_based_gamma(value)
            mask = gammas.gt(thre).float().cuda()
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            num_pruned += mask.shape[0] - torch.sum(mask)
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')

    pruned_ratio = num_pruned / num_total
    print('Pre-processing Successful! Pruned ratio: ', pruned_ratio)

    acc = test(model, test_loader)

    # new model
    newmodel = vgg(dataset=args.dataset, cfg=cfg)
    if args.cuda:
        newmodel.cuda()
    print(cfg)

    num_parameters = sum([param.nelement() for param in newmodel.parameters()])
    savepath = os.path.join(args.save, "prune.txt")
    with open(savepath, "w") as fp:
        fp.write("Configuration: \n{}\n".format(cfg))
        fp.write("Prune ratio: {:.4f}\n".format(pruned_ratio))
        fp.write("Number of parameters: {}\n".format(num_parameters))
        fp.write("Origin Model accuracy: {:.4f}\n".format(best_prec1))
        fp.write("Test Pruned Model accuracy: {:.4f}".format(acc))

    layer_id_in_cfg = 0
    start_mask = torch.ones(3)  # initially three channels
    end_mask = cfg_mask[layer_id_in_cfg]  # the first mask in the output of the next layer (the input of next layer)
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()  # Calculate the average of the data so far
            m1.running_var = m0.running_var[idx1.tolist()].clone()  # Calculate the variance of the data so far
            layer_id_in_cfg += 1  # next cfg
            start_mask = end_mask.clone()  # next start mask
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]  # next end mask
        elif isinstance(m0, nn.Conv2d):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))  # shape: () convert to (1,)
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

    torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))

    print(newmodel)
    model = newmodel
    test(model, test_loader)
