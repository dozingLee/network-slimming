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
    
    (1) Prune Sparsity VGG19 with 70% proportion for cifar10 (Test accuracy: 19.17%) 12.96
    python vggprune_attention.py --dataset cifar10 --depth 19 --percent 0.7 
        --model ./logs/sparsity_vgg19_cifar10_s_1e-4/model_best.pth.tar --save ./logs/attention_prune_vgg19_percent_0.7
    
    (2) Prune Sparsity VGG19 with 50% proportion for cifar10 (Test accuracy: 93.47%) !!!!
    python vggprune.py --dataset cifar10 --depth 19 --percent 0.5 
        --model ./logs/sparsity_vgg19_cifar10_s_1e-4/model_best.pth.tar --save ./logs/prune_vgg19_percent_0.5
        
    (3) Prune Sparsity VGG19 with 30% proportion for cifar10 (Test accuracy: 93.47%) !!!!
    python vggprune.py --dataset cifar10 --depth 19 --percent 0.3 
        --model ./logs/sparsity_vgg19_cifar10_s_1e-4/model_best.pth.tar --save ./logs/prune_vgg19_percent_0.3
        

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

# Model
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
print('Origin model: \r\n', model)

# Dataset
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

# Algorithm
def attention_based_gramma(weight_data):
    # 1. A: resize and absolute
    d1, d2 = weight_data.shape[0], weight_data.shape[1]
    A = weight_data.view(d1, d2, -1).abs()
    c, h, w = A.shape

    # 2. F(A): sum of values along the channel direction
    FA = torch.zeros(h, w).cuda()
    for i in range(c):
        FA.add_(A[i])

    # 3. ||F(A)||^2: square of two norm
    FA_s = torch.linalg.norm(FA)

    # 4. F(A) / ||F(A)||^2: normalize weight data
    F_all = FA / FA_s

    # 5. F(Aj) / ||F(Aj)||^2 & gamma = ∑ | F(A)/||F(A)||^2 - F(Aj)/||F(Aj)||^2 |: pruning standard
    gammas = torch.zeros(c)
    for j in range(c):
        Aj = A[j]
        FAj = FA - Aj
        FAj_s = torch.linalg.norm(FAj)
        Fj = FAj / FAj_s
        gamma_j = (F_all - Fj).abs().sum()
        gammas[j] = gamma_j

    return gammas

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
for idx, m in enumerate(model.features):
    one_batch = m(one_batch)
    if isinstance(m, nn.ReLU):
        value = one_batch.clone().squeeze(0)
        size = value.shape[0]
        gammas = attention_based_gramma(value)
        gamma_list[index:(index+size)] = gammas.clone()
        index += size

# threshold
y, i = torch.sort(gamma_list)
thre_idx = int(num_total * args.percent)
thre = gamma_list[thre_idx]

# Pruned
num_pruned = 0
cfg = []
cfg_mask = []
one_batch = data1.clone()
for k, m in enumerate(model.modules()):
    one_batch = m(one_batch)
    if isinstance(m, nn.BatchNorm2d):
        value = nn.ReLU(one_batch.clone())
        mask = value.gt(thre).float().cuda()
        num_pruned += mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))
    elif isinstance(m, nn.MaxPool2d):
        cfg.append('M')

pruned_ratio = num_pruned / num_total
print('Pre-processing Successful! Pruned ratio: ', pruned_ratio)


def test(model, dataloader):
    model.eval()
    correct = 0
    for data, target in dataloader:
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
    acc = test(model, test_loader)

    # new model
    newmodel = vgg(dataset=args.dataset, cfg=cfg)
    if args.cuda:
        newmodel.cuda()
    print(cfg)

    num_parameters = sum([param.nelement() for param in newmodel.parameters()])
    savepath = os.path.join(args.save, "prune.txt")
    with open(savepath, "w") as fp:
        fp.write("Configuration: \n" + str(cfg) + "\n")
        fp.write("Number of parameters: \n" + str(num_parameters) + "\n")
        fp.write("Test accuracy: \n" + str(acc))

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
            m1.running_var = m0.running_var[idx1.tolist()].clone()    # Calculate the variance of the data so far
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
