# -*- coding: utf-8 -*-

import numpy as np
import torch

from common.utils import wrap
from common.quaternion import qrot, qinverse

# camera frame normalization
# 将图片标准化到[-1, 1]之间，训练用到的2D pose是在w=1000, h=1002的图片上训练出来的,将像素坐标[0,w]归一化到[-1,1]
# 2D coordinates, N * 2，标准化之后可以使算法更快收敛，提高算法精度
def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2  # 2D coordinates

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]


def image_coordinates(X, w, h): # [0, w], 像素坐标系
    assert X.shape[-1] == 2

    # Reverse camera frame normalization
    return (X + [1, h / w]) * w / 2


def camera_to_world(X, R, t):
    return wrap(qrot, False, np.tile(R, X.shape[:-1] + (1, )), X) + t

# 世界坐标系到相机坐标系的变换
# 世界坐标系到相机坐标系的转换，即计算出以相机为坐标原点时，所有顶点在相机坐标系中的坐标
# 将相机从原点变换到世界坐标系时，相机的变换为Mr*Mt,Mr表示旋转矩阵，Mt表示平移矩阵，
# 那么以相机为原点时，可以看做是整个世界的物体相对于相机做了逆变换，对于物体来讲， 物体的变换就是(Mr * Mt)^(-1) = Mt^(-1)*Mr^(-1)
# 即先向相反的方向平移，再反方向旋转
def world_to_camera(X, R, t):
    Rt = wrap(qinverse, False, R)  # R: (1, 4)
    return wrap(qrot, False, np.tile(Rt, X.shape[:-1] + (1,)), X - t)  # reverse rotate and translate

# X.shape[:-1]表示(16, ) ==> 取关节点个数
# np.tile(Rt, (16, 1)) ==> (16, 4)
# 可以进行element-wise变换操作

# def world_to_camera(X, R, t):
#     Rt = wrap(qinverse, False, R)
#     Q = np.tile(Rt, X.shape[:-1] + (1,))    # Q = np.tile(Rt, (*X.shape[:-1], 1))
#     V = X - t
#     return wrap(qrot, False, Q, V)



def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6m camera projection function.
    :X: 3D joints in camera space to transform, (N, *, 3)
    :camera_params: intrinsic parameters (N, 2+2+3+2=9), 即透视投影矩阵
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)  # (N, 1, 9)

    f = camera_params[..., :2]  # 第0,1个参数是焦距
    c = camera_params[..., 2:4]  # 第2,3个参数是投影中心，主点(主光轴在物理成像平面上的焦点)
    k = camera_params[..., 4:7]  # 第4,5,6个参数是径向畸变系数k1, k2, k3
    p = camera_params[..., 7:]  # 第7,8个参数是切向畸变系数p1, p2

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)  # xc / zc, yc /zc, 再“夹紧”到-1 ～ 1, (N, *, 1)

    r2 = torch.sum(XX[..., :2] ** 2, dim=len(XX.shape) - 1, keepdim=True)  # (xc/zc)**2 + (yc/zc)**2, (N, *, 1)

    radial = 1 + torch.sum(k * torch.cat((r2, r2 ** 2, r2 ** 3), dim=len(r2.shape) - 1), dim=len(r2.shape) - 1,
                           keepdim=True)  # 1+k*(r^2+k2*r^4+k3*r^6), 先按列拼接 -> (N, *, 3) 再乘以k -> (N, *, 3)
    # 再按列求和 -> (N, *, 1), 在实际计算过程中，如果考虑太多高阶的畸变参数，会导致标定求解的不稳定。
    tan = torch.sum(p * XX, dim=len(XX.shape) - 1, keepdim=True)  # p1 * (xc /zc) + p2 * (yc / zc), (N, *, 1)

    XXX = XX * (radial + tan) + p * r2  # (N, *, 2)

    return f * XXX + c


def project_to_2d_linear(X, camera_params):
    """
    Project 3D points to 2D using only linear parameters (focal length and principal point).

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameters (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)  # x_c / z_c, y_c / z_c

    return f * XX + c  # f * (x_c / z_c) + cx, f * (y_c / z_c) + cy

# 相机坐标系下，一个空间点的三维坐标到其对应像点到像素坐标到转换关系
# u = fx * (xc /zc) + cx
# v = fy * (yc /zc) + cx


def camera_project(X, camera_params):
    """
    Project 3D points to 2D using only linear parameters (focal length and principal point).

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameters (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 6
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)  # x_c / z_c, y_c / z_c

    return f * XX + c  # f * (x_c / z_c) + cx, f * (y_c / z_c) + cy

