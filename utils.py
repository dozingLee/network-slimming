import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import numpy as np
import random
from torch.backends import cudnn
import torch.nn.functional as F

from models import vgg


def init_seeds(seed, acce=False):
    """
    :param seed: random seed
    :param acce: True or False

    manual_seed():  set the seed for generating random numbers
    cuda.manual_seed(): set the seed for generating random numbers for the current GPU
    cuda.manual_seed_all(): set the seed for generating random numbers on all GPUs
    random & np.random

    cudnn.deterministic: The convolution operation is optimized in cudnn, and the accuracy
                        is sacrificed in exchange for the computational efficiency.
    cuda.benchmark: If the accuracy requirement  is not very high, in fact, it is not recommended
                        to modify, because it will reduce the computational efficiency.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    if acce:
        cudnn.deterministic = True
        cudnn.benchmark = False


def generate_vgg_model(dataset, depth, model_path, cuda_available):
    model = vgg(dataset=dataset, depth=depth)
    if cuda_available:
        model.cuda()

    best_prec1 = 0.
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:.4f}"
              .format(model_path, checkpoint['epoch'], best_prec1))
    else:
        raise ValueError("=> no model file found at '{}'".format(model_path))
    return model, best_prec1


def get_test_loader(dataset, test_batch_size, num_thread, cuda_available):
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_loader = DataLoader(getattr(datasets, dataset.upper())(
        'data.{}'.format(dataset), train=False, download=True, transform=test_transforms),
        batch_size=test_batch_size, shuffle=True, num_workers=num_thread, pin_memory=cuda_available)
    return test_loader


def get_train_loader(dataset, train_batch_size, num_thread, cuda_available):
    train_transforms = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_loader = DataLoader(getattr(datasets, dataset.upper())(
        'data.{}'.format(dataset), train=True, download=True, transform=train_transforms),
        batch_size=train_batch_size, shuffle=True, num_workers=num_thread, pin_memory=cuda_available)
    return train_loader


def get_dataset_loaders(dataset, train_batch_size, test_batch_size, num_thread, cuda_available):
    train_loader = get_train_loader(dataset, train_batch_size, num_thread, cuda_available)
    test_loader = get_test_loader(dataset, test_batch_size, num_thread, cuda_available)
    return train_loader, test_loader


def resume_model(resume_file, model, optimizer):
    if not os.path.isfile(resume_file):
        raise ValueError("Resume model file is not found at '{}'".format(resume_file))
    print("=> loading checkpoint '{}'".format(resume_file))
    checkpoint = torch.load(resume_file)
    start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
          .format(resume_file, start_epoch, best_prec1))
    return model, optimizer, start_epoch, best_prec1


def save_checkpoint(state, is_best, save_path):
    check_point_path = os.path.join(save_path, 'checkpoint.pth.tar')
    model_best_path = os.path.join(save_path, 'model_best.pth.tar')
    torch.save(state, check_point_path)
    if is_best:
        shutil.copyfile(check_point_path, model_best_path)


def visualization_record(save_path):
    data = pd.read_csv(os.path.join(save_path, 'record.csv'))
    line_loss, = plt.plot(data['loss'], 'r-')
    line_prec, = plt.plot(data['prec'], 'b-')
    plt.legend([line_loss, line_prec], ['loss', 'accuracy'], loc='upper right')
    plt.ylabel('value', fontsize=12)
    plt.xlabel('epoch', fontsize=12)
    plt.title('Train loss and accuracy (best_prec1: {})'.format(max(data['prec'])), fontsize=14)
    plt.savefig(os.path.join(save_path, "record train loss.png"))
    print('Save the training loss and accuracy successfully.')


def at(x):
    """
    :param x: input data ∈ [B, C, H, W]
        B: batch size
        C: channel size
        H: feature map height
        W: feature map width

        pow(2) -> mean(1) -> view(B, -1) -> normalize
            pow(2): [B, C, H, W] square each data
            mean(1): [B, 1, H, W] 1 denotes averaging channel direction
            view(B, -1): [B, 1 × H × W] batch direction
            normalize: [B, H × W] N(0, 1) for each data in the batch

    :return: [B, H × W]
    """
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def at_loss(x, y):
    """
    :param x: input data1 [B, C, H, W]
    :param y: input data2 [B, C, H, W]

        [B, C, H, W] -> [B, H × W] -> []
        1. at(x), at(y)
        2. mean()

    :return: []
    """
    return (at(x) - at(y)).pow(2).mean()


def bytes_to_human(n):
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if n >= prefix[s]:
            value = float(n) / prefix[s]
            return '%.1f%s' % (value, s)
    return '%sB' % n