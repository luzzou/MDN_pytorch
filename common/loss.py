# -*- coding: utf-8 -*-

import torch
import numpy as np


def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as 'Protocol #1' in many papers
    """

    assert predicted.shape == target.shape   # (batch_size, 16,3)
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))  # 对输入的指定维度求范数，默认为L2


def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after alignment (scale, rotation, and translation), similarity_transform
    often referred to as "Protocol #2" in many papers.
    alignment: 将两个不同的形状进行归一化的过程，使一个形状尽可能地贴近另一个形状
    """
    assert predicted.shape == target.shape    # x: (batch_size, num_points, num_dims)


    # X: array NxM of targets, with N number of points and M point dimensionality
    # Y: array NxM of inputs
    muX = np.mean(target, axis=1, keepdims=True)  # (batch_size, 1, num_dims), 求出关节点x,y,z方向的坐标平均值
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target-muX   # 去中心化, 消除平移的影响
    Y0 = predicted -muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))  # 根据去中心化数据，计算每个样本的重心
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))  # (batch_size, 1,1)

    X0 /= normX
    Y0 /= normY  # 归一化，消除缩放的影响, (batch_size, num_points, num_dims)

    H = np.matmul(X0.transpose(0, 2, 1), Y0)  # (batch_size, num_dims. num_dims)
    U, s, Vt = np.linalg.svd(H) # 若H为(M,N), u大小为(M,M)，s大小为(M,N)，v大小为(N,N)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))  # (batch_size, num_dims. num_dims)

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    # 旋转变换矩阵的行列式为 +1
    # 旋转变换是正交变换的一种。而正交变换分为两类：第一类是旋转变换，第二类是镜面反射。正交变换的矩阵的行列式等于 1 或-1
    # 规定行列式等于 1 的正交变换称为旋转变换，行列式等于-1 正交变换称为镜面反射
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))  # 增加一个维度, (batch_size, 1),将每个样本的|R|转换为正负1
    V[:,:,-1] *= sign_detR  # (,)
    s[:,-1] *= sign_detR.flatten()  # 展开成一维数组
    R = np.matmul(V, U.transpose(0, 2, 1))  # rotation

    trance = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = trance * normX / normY   # scale
    t = muX - a * np.matmul(muY, R)  # translation

    # perform a rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) -1))
    # return torch.mean(torch.norm(predicted_aligned - target, dim=len(target.shape) -1))

# https://github.com/una-dinosauria/3d-pose-baseline/blob/master/src/procrustes.py


def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape   # x: (batch_size, num_points, num_dims)

    norm_predicted = torch.mean(torch.sum(predicted **2, dim=2, keepdim=True), dim=1, keepdim=True)
    norm_target = torch.mean(torch.sum(target * predicted, dim=2, keepdim=True), dim=1, keepdim=True)
    scale = norm_target/norm_predicted
    return mpjpe(scale * predicted, target)


def cam_loss(predicted):
    """loss function to enforce a weak perspective camera"""

    k = predicted.view(-1, 2, 3)  # (64, 2, 3), m是相机网络的输出,  先对predicted进行PMPJPE矫正
    k_sq = torch.matmul(k, k.transpose(2, 1))  # (64, 2, 2)
    trace = torch.rand(k_sq.size(0)).cuda()
    for i in range(k_sq.size(0)):
        trace_element = k_sq[i, ...].trace()
        trace[i] = trace_element

    loss_mat = (2/ trace).view(-1, 1, 1) * k_sq - torch.eye(2).cuda()
    loss_cam = torch.norm(loss_mat, p=1, dim=[1, 2]).mean()  # F1 norm

    return loss_cam
