# -*- coding: utf-8 -*-

import torch

# 定义3D旋转的四元数形式
# https://blog.molecular-matters.com/2013/05/24/a-faster-quaternion-vector-multiplication/
def qrot(q,v):
    """
    Rotate vector(s) v about the rotation descried by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v
    Where * denotes any number of dimensions.
    :return: a tensor of shape (*, 3)
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[: -1] == v.shape[: -1]  # (batch_size, num_points)

    # qvec = q[..., 1:]  # 四元数的向量部分. 例如q是一个形状为(5,10,3)的tensor,q[..., :1]的形状是(5,10,1) q[:, :1]的形状是(5,1,3)
    # w = q[..., :1]  # 四元数的标量部分
    # t = 2 * torch.cross(qvec, v)
    # return v + w * t + torch.cross(qvec, t)

    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=(len(q.shape) -1) )  # uv = np.cross(qvec, v)
    uuv = torch.cross(qvec, uv, dim=(len(q.shape) - 1))  # uuv = np.cross(qvec, uv)

    return v + 2 * (q[..., :1] * uv + uuv)


def qinverse(q, inplace=False):  # 求四元数的共轭
    # We assume the quaternion to be normalized
    if inplace:
        q[..., 1:] *= -1
        return q
    else:
        w = q[..., :1]
        xyz = q[..., 1:]
        return torch.cat((w, -xyz), dim=len(q.shape) - 1)  # np.hstack(w, -xyz), 按列顺序将数组堆叠起来