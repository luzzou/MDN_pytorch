# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn.functional as F

def Dirichlet_loss(parameters, c, m, prior):
    '''
    add dirichlet Conjugate prior to the loss function to prevent all data fitting into single kernel
    '''

    components = parameters.view(-1, c+2, m)   # parameters: (64, 235), m: 5, c: 15*3, components: (64, 47, 5)
    alpha = components[:, c + 1, :]
    alpha = torch.clamp(alpha, 1e-8, 1.)
    loss = torch.sum((prior-1.0)*torch.log(alpha),dim=1)
    res = -torch.mean(loss)

    return res



def log_sum_exp(x, dim=None):
    """Log-sum-exp trick implementation"""

    x_max = x.topk(1, dim=1)[0]
    x = x - x_max
    a = torch.sum(torch.exp(x), dim=1, keepdim=True)
    out = torch.log(a) + x_max

    return out


def mean_log_Gaussian_like(y_true, parameters,c,m ):
    """Mean Log Gaussian Likelihood distribution
    y_truth: ground truth 3d pose
    parameters: output of hypotheses generator, which conclude the mean, variance and mixture coefficient of the mixture model
    c: dimension of 3d pose
    m: number of kernels
    """


    components = parameters.view(-1, c + 2, m)  # c: 15*3, m: num_models
    mu = components[:, :c, :]  # (64, 45, num_models)
    sigma = components[:, c, :]

    sigma = torch.clamp(sigma, 1e-2,1e2)
    # sigma = torch.clamp(sigma, 1e-2, 1e2)  # sigma = torch.clamp(sigma, 1e-2, 1e1), OK, 1e-2, 2e2也可以，找一个效果最好的点

    alpha = components[:, c + 1, :]
    alpha = torch.clamp(alpha, 1e-8, 1.)

    # gaussian distribution
    normal_part = torch.log(alpha) - 0.5 * c * torch.log((torch.tensor(2 * np.pi)))- c * torch.log(sigma)
    exponent = normal_part - torch.sum((y_true.unsqueeze(2)- mu) ** 2, dim=1)  / (2.0 * (sigma) ** 2.0)

    log_gauss = log_sum_exp(exponent, dim=1)  # (batch_size, 1)
    res = - torch.mean(log_gauss)
    return res


