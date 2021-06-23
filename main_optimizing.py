from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import models
import utils


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
parser.add_argument('--not-init-weight', action='store_true', default=False,
                    help='not initialize model weight, working while refining the pruned model (default: False)')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

utils.init_seeds(args.seed)


def generate_model(args):
    """
    :param args: all parameters
        if args.refine:
            pruned model must have 'cfg'
            if args.not_init_weight:
                pruned model must have `state_dict`
        else:
            model automatically init weight
    """
    model_cfg = []
    if args.refine:
        file = torch.load(args.refine)
        if 'cfg' not in file:
            raise ValueError('Refine pruned model, but the pruned model file is not including `cfg`.')
        if 'state_dict' not in file and args.not_init_weight:
            raise ValueError('Not init weight, but the pruned model file is not including `state_dict`.')

        model_cfg = file['cfg']
        model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=model_cfg)

        if args.not_init_weight:
            model.load_state_dict(file['state_dict'])
            print('Pruned model loads important weight successfully!')
    else:
        model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

    if args.cuda:
        model.cuda()

    return model, model_cfg


# additional sub_gradient descent on the sparsity-induced penalty term
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

        if args.sr:
            updateBN()  # sparsity regularization

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

    length = len(test_loader.dataset)
    test_loss /= length
    test_prec = float(correct) / length
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f})\n'.format(
        test_loss, correct, length, test_prec))
    return test_prec, test_loss


if __name__ == '__main__':
    # ==== Dataset ====
    train_loader, test_loader = utils.get_dataset_loaders(
        args.dataset, args.batch_size, args.test_batch_size, args.nthread, args.cuda)

    # ==== Model ====
    model, model_cfg = generate_model(args)

    # ==== Optimizer ====
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # ---- record data ----
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    start_epoch, best_prec1 = 0, 0.
    record_file = os.path.join(args.save, 'record.csv')

    if args.resume:
        model, optimizer, start_epoch, best_prec1 = utils.resume_model(args.resume, model, optimizer)
        f = open(record_file, 'a+')
    else:
        f = open(record_file, 'w+')
        f.write('epoch,loss,prec\n')

    # ====== Training ======
    for epoch in range(start_epoch, args.epochs):
        if epoch in [args.epochs * 0.5, args.epochs * 0.75]:  # decent the learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        train(epoch)                    # process train
        prec1, loss1 = test()           # process test
        is_best = prec1 > best_prec1    # save the best
        best_prec1 = max(prec1, best_prec1)

        check_point_dict = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }
        if model_cfg:
            check_point_dict['cfg'] = model_cfg
        utils.save_checkpoint(check_point_dict, is_best, args.save)

        f.write('{},{:.4f},{:.4f}\n'.format(epoch, loss1, prec1))

    utils.visualization_record(args.save)
    print("Best accuracy: {:.4f}".format(best_prec1))