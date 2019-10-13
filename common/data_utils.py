# -*- coding: utf-8 -*-

import numpy as np
import torch

from functools import reduce
from torch.utils.data import Dataset
from common.camera import normalize_screen_coordinates, world_to_camera


# 2D poses are scaled and normalized according to the image resolution and normalized to [-1, 1]
def create_2d_data(data_path, dataset):
    keypoints = np.load(data_path, allow_pickle=True, encoding='latin1')
    keypoints = keypoints['positions_2d'].item()

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]  # kps, shape (*, 16, 2), kps和kps[..., :2]是一样的
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

    return keypoints


def read_3d_data(dataset):
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, :] -= pos_3d[:, :1]  # Remove global offset, 所有action的16个点，都减去了第一个点的坐标值
                positions_3d.append(pos_3d)

            anim['positions_3d'] = positions_3d

    return dataset



class PoseGenerator(Dataset):
    def __init__(self, poses_3d, poses_2d, actions):
        assert poses_3d is not None


        self._poses_3d = np.concatenate(poses_3d)  # 本来是600
        self._poses_2d = np.concatenate(poses_2d)
        self._actions = reduce(lambda x, y: x + y, actions)   # actions = ['a','b','b','2','f','f'] ==> actions = 'abb2ff'

        assert self._poses_3d.shape[0] == self._poses_2d.shape[0] and self._poses_3d.shape[0] == len(self._actions)
        print('Generating {} poses...'.format(len(self._actions)))



    def __getitem__(self, index):
        out_pose_3d = self._poses_3d[index]
        out_pose_2d = self._poses_2d[index]
        out_action = self._actions[index]

        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        out_pose_2d = torch.from_numpy(out_pose_2d).float()

        return out_pose_3d, out_pose_2d, out_action

    def __len__(self):
        return len(self._actions)