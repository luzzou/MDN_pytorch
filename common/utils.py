# -*- coding: utf-8 -*-

import os
import torch
import hashlib
import numpy as np

# Use an average meter to compute the global loss
class AverageMeter(object):
    """Compute and store the average and current values"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def lr_decay(optimizer, step, lr, decay_step, gamma):
    lr = lr * gamma ** (step / decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_ckpt(state, ckpt_path, suffix=None):
    if suffix is None:
        suffix = 'epoch_{:04d}'.format(state['epoch'])
    file_path = os.path.join(ckpt_path, 'ckpt_{}.pth.tar'.format(suffix))
    torch.save(state, file_path)


def wrap(func, unsqueeze, *args):
    """
    Wrap a torch function so it can be called with Numpy arrays.
    Input and return types are seamlessly converted.
    """

    # 将输入的numpy数据先转换为torch.tensor,对tensor进行相应的操作
    # convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                arg[i] = arg[i].unsqueeze(0)  # 对维度进行扩充，在指定位置上增加一个维度为1的维度

    result = func(*args)

    # 将生成的tensor转换为numpy之后再返回
    # convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)  # 压缩维度为1的维度
                result[i] = res.numpy()

            return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')  # 'a' for appending

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()  # 刷新缓冲区

    def close(self):
        self.f.close()  # 关闭文件


def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2**32 - 1) * (max_value - min_value)) + min_value