import torch
import numpy as np
import torch.nn as nn

import utils
from models import channel_selection
from models.preresnet import Bottleneck


def generate_new_vgg_model(model, new_model, cfg_mask):
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
            # print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
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
    print('Generate new VGG Model successfully！')
    return new_model


def generate_new_densenet_model(model, new_model, cfg_mask):
    old_modules, new_modules = list(model.modules()), list(new_model.modules())
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    first_conv = True

    for layer_id in range(len(old_modules)):
        m0 = old_modules[layer_id]
        m1 = new_modules[layer_id]
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))

            if isinstance(old_modules[layer_id + 1], channel_selection):
                # If the next layer is the channel selection layer,
                # then the current batch normalization layer won't be pruned.
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                # We need to set the mask parameter `indexes` for the channel selection layer.
                m2 = new_modules[layer_id + 1]
                m2.indexes.data.zero_()
                m2.indexes.data[idx1.tolist()] = 1.0

                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
                continue

        elif isinstance(m0, nn.Conv2d):
            if first_conv:
                # We don't change the first convolution layer.
                m1.weight.data = m0.weight.data.clone()
                first_conv = False
                continue
            if isinstance(old_modules[layer_id - 1], channel_selection):
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))

                # If the last layer is channel selection layer,
                # then we don't change the number of output channels of the current
                # convolutional layer.
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                m1.weight.data = w1.clone()
                continue

        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))

            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()
    print('Generate new DenseNet Model successfully！')
    return new_model


def generate_new_resnet_model(model, new_model, cfg_mask):
    old_modules = list(model.modules())
    new_modules = list(new_model.modules())
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    conv_count = 0

    for layer_id in range(len(old_modules)):
        m0, m1 = old_modules[layer_id], new_modules[layer_id]
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            if isinstance(old_modules[layer_id + 1], channel_selection):
                # If the next layer is the channel selection layer,
                # then the current batchnorm 2d layer won't be pruned.
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                # We need to set the channel selection layer.
                m2 = new_modules[layer_id + 1]
                m2.indexes.data.zero_()
                m2.indexes.data[idx1.tolist()] = 1.0

                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
            else:
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            if conv_count == 0:
                m1.weight.data = m0.weight.data.clone()
                conv_count += 1
                continue
            if isinstance(old_modules[layer_id-1], channel_selection) or isinstance(old_modules[layer_id-1], nn.BatchNorm2d):
                # This convers the convolutions in the residual block.
                # The convolutions are either after the channel selection layer or after the batch normalization layer.
                conv_count += 1
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                # print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()

                # If the current convolution is not the last convolution in the residual block, then we can change the
                # number of output channels. Currently we use `conv_count` to detect whether it is such convolution.
                if conv_count % 3 != 1:
                    w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
                continue

            # We need to consider the case where there are downsampling convolutions.
            # For these convolutions, we just copy the weights.
            m1.weight.data = m0.weight.data.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))

            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()
    print('Generate new RseNet Model successfully！')
    return new_model


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


def bn_prune_model(model, percent, cuda_available):
    threshold, thre_idx = bn_threshold(model, percent)
    if cuda_available:
        threshold = threshold.cuda()
    num_pruned, num_total = 0, 0
    cfg, cfg_mask = [], []
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(threshold).float()        # if value > threshold: True, else: False
            if torch.sum(mask) == 0:            # avoid pruning channel to zero
                mask[0] = 1.0                   # by default, the first channel is treated as a reserved value
            m.weight.data.mul_(mask)            # pruned weight
            m.bias.data.mul_(mask)              # pruned bias
            cfg.append(int(torch.sum(mask)))    # int: transfer tensor into numpy
            cfg_mask.append(mask.clone())
            num_total += mask.shape[0]
            num_pruned += mask.shape[0] - torch.sum(mask)  # num of pruned
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'
                  .format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')
    pruned_ratio = num_pruned / num_total
    print('Preprocess Successfully! Pruned ratio: {}'.format(pruned_ratio))
    print('Pruned cfg: {}'.format(cfg))
    return model, cfg, cfg_mask, pruned_ratio


def at_vgg_threshold(model, percent, one_batch):
    """
    :param model: model cuda
    :param percent: [0,1]
    :param one_batch: [batch_size, 3, 32, 32]
    :return threshold and threshold index
    """
    num_total = 0
    for m in model.feature:
        if isinstance(m, nn.BatchNorm2d):
            num_total += m.weight.data.shape[0]
    gamma_list = torch.zeros(num_total)

    index = 0
    data = one_batch.clone()
    for idx, m in enumerate(model.feature):
        data = m(data)
        if isinstance(m, nn.BatchNorm2d):
            value = data.clone()
            gamma = utils.gammas(value)
            size = value.shape[1]
            gamma_list[index:(index + size)] = gamma.clone()
            index += size

    y, i = torch.sort(gamma_list)
    threshold_index = int(num_total * percent)
    threshold = y[threshold_index]
    return threshold, threshold_index


def at_vgg_prune_model(model, percent, one_batch, cuda_available):
    threshold, thre_idx = at_vgg_threshold(model, percent, one_batch)
    if cuda_available:
        threshold = threshold.cuda()
    num_pruned, num_total = 0, 0
    cfg, cfg_mask = [], []
    data = one_batch.clone()
    for k, m in enumerate(model.feature):
        data = m(data)
        if isinstance(m, nn.BatchNorm2d):
            value = data.clone()
            gammas = utils.gammas(value)
            if cuda_available:
                gammas = gammas.cuda()
            mask = gammas.gt(threshold).float()
            if torch.sum(mask) == 0:
                mask[0] = 1.0
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            num_total += mask.shape[0]
            num_pruned += mask.shape[0] - torch.sum(mask)
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')
    pruned_ratio = num_pruned / num_total
    print('Preprocess Successfully! Pruned ratio: {}'.format(pruned_ratio))
    print('Pruned cfg: {}'.format(cfg))
    return model, cfg, cfg_mask, pruned_ratio


# def at_resnet_threshold_2(model, percent, one_batch):
#     num_total = 0
#     for m in model.modules():
#         if isinstance(m, nn.BatchNorm2d):
#             num_total += m.weight.data.shape[0]
#     gamma_list = torch.zeros(num_total)
#
#     index = 0
#     data = one_batch.clone()
#     for idx, m in enumerate(model.modules()):
#         if isinstance(m, nn.Conv2d) or isinstance(m, nn.ReLU) or isinstance(m, channel_selection):
#             data = m(data)
#         elif isinstance(m, nn.BatchNorm2d):
#             data = m(data)
#             value = data.clone()
#             gamma = utils.gammas(value)
#             size = value.shape[1]
#             gamma_list[index:(index+size)] = gamma.clone()
#             index += size
#     y, i = torch.sort(gamma_list)
#     threshold_index = int(num_total * percent)
#     threshold = y[threshold_index]
#     print("AT (attention transfer) 2 resnet index:{}, threshold: {}".format(threshold_index, threshold))
#     return threshold, threshold_index


def at_resnet_threshold(model, percent, one_batch):
    """
    :param model: model cuda
    :param percent: [0,1]
    :param one_batch: [batch_size, 3, 32, 32]
    :return threshold and threshold index
    """
    num_total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            num_total += m.weight.data.shape[0]
    gamma_list = torch.zeros(num_total)

    index = 0
    data = one_batch.clone()
    for idx, m in enumerate(model.feature):         # modules (-classifier) -> feature
        if isinstance(m, nn.Sequential):            # feature (-Conv2d) -> layers
            for j, n in enumerate(m.children()):    # layers -> layer (many Bottlenecks)
                data, bn_value = n.forward_bn(data)
                for i, value in enumerate(bn_value):    # 3 batchNorm2ds
                    gamma = utils.gammas(value.clone())
                    size = value.shape[1]
                    gamma_list[index:(index+size)] = gamma.clone()
                    index += size
        else:
            data = m(data)
            if isinstance(m, nn.BatchNorm2d):
                gamma = utils.gammas(data.clone())
                size = data.shape[1]
                gamma_list[index:(index+size)] = gamma.clone()
                index += size

    y, i = torch.sort(gamma_list)
    threshold_index = int(num_total * percent)
    threshold = y[threshold_index]
    print("AT (attention transfer) resnet index:{}, threshold: {}".format(threshold_index, threshold))
    return threshold, threshold_index


def at_resnet_prune_model(model, percent, one_batch, cuda_available):
    threshold, thre_idx = at_resnet_threshold(model, percent, one_batch)
    if cuda_available:
        threshold = threshold.cuda()
    num_pruned, num_total = 0, 0
    cfg, cfg_mask = [], []
    data = one_batch.clone()
    for k, m in enumerate(model.feature):
        if isinstance(m, nn.Sequential):
            for j, n in enumerate(m.children()):
                data, bn_value = n.forward_bn(data)
                for i, value in enumerate(bn_value):    # 3 batchNorm2ds
                    gamma = utils.gammas(value.clone())
                    if cuda_available:
                        gamma = gamma.cuda()
                    mask = gamma.gt(threshold).float()
                    if torch.sum(mask) == 0:
                        mask[0] = 1.0
                    n.mask(i, mask)
                    cfg.append(int(torch.sum(mask)))
                    cfg_mask.append(mask.clone())
                    num_total += mask.shape[0]
                    num_pruned += mask.shape[0] - torch.sum(mask)
                    print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                          format(k, mask.shape[0], int(torch.sum(mask))))
        else:
            data = m(data)
            if isinstance(m, nn.BatchNorm2d):
                gamma = utils.gammas(data.clone())
                if cuda_available:
                    gamma = gamma.cuda()
                mask = gamma.gt(threshold).float()
                if torch.sum(mask) == 0:
                    mask[0] = 1.0
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                cfg.append(int(torch.sum(mask)))
                cfg_mask.append(mask.clone())
                num_total += mask.shape[0]
                num_pruned += mask.shape[0] - torch.sum(mask)
                print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                      format(k, mask.shape[0], int(torch.sum(mask))))

    pruned_ratio = num_pruned / num_total
    print('Preprocess Successfully! Pruned ratio: {}'.format(pruned_ratio))
    print('Pruned cfg: {}'.format(cfg))
    return model, cfg, cfg_mask, pruned_ratio