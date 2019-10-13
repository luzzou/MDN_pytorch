# -*- coding: utf-8 -*-

import time
import argparse
import numpy as np
import os
import os.path as path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from progress.bar import Bar
from common.utils import AverageMeter
from common.data_utils import read_3d_data, create_2d_data
from common.data_utils import PoseGenerator
from common.h36m_dataset import Human36mDataset
from common.loss import mpjpe, p_mpjpe, n_mpjpe
from common.camera import camera_to_world, image_coordinates
from common.visualization import render_animation
from common.arguments import parse_args

from models.mdn import LinearModel, weight_init

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main(args):
    print('==> Using settings {}'.format(args))

    print('==> Loading dataset...')
    dataset_path = path.join('data', 'data_3d_' + args.dataset + '.npz')
    dataset = Human36mDataset(dataset_path)

    print('==> Preparing data...')
    dataset = read_3d_data(dataset)

    print('==> Loading 2D detections...')
    keypoints = create_2d_data(path.join('data', 'data_2d_' + args.dataset + '_' + args.keypoints + '.npz'), dataset)

    cudnn.benchmark = True
    device = torch.device("cuda")

    # Create model
    print("==> Creating model...")

    num_joints = dataset.skeleton().num_joints()
    model_pos = LinearModel(num_joints * 2, (num_joints - 1) * 3).to(device)
    model_pos.apply(weight_init)

    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_pos.parameters()) / 1000000.0))

    # Resume from a checkpoint
    ckpt_path = args.evaluate

    if path.isfile(ckpt_path):
        print("==> Loading checkpoint '{}'".format(ckpt_path))
        ckpt = torch.load(ckpt_path)
        start_epoch = ckpt['epoch']
        error_best = ckpt['error']
        model_pos.load_state_dict(ckpt['state_dict'])
        print("==> Loaded checkpoint (Epoch: {} | Error: {})".format(start_epoch, error_best))
    else:
        raise RuntimeError("==> No checkpoint found at '{}'".format(ckpt_path))

    print('==> Rendering...')

    poses_2d = keypoints[args.viz_subject][args.viz_action]
    out_poses_2d = poses_2d[args.viz_camera]
    out_actions = [args.viz_camera] * out_poses_2d.shape[0]

    poses_3d = dataset[args.viz_subject][args.viz_action]['positions_3d']
    assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
    out_poses_3d = poses_3d[args.viz_camera]

    ground_truth = dataset[args.viz_subject][args.viz_action]['positions_3d'][args.viz_camera].copy()

    input_keypoints = out_poses_2d.copy()
    render_loader = DataLoader(PoseGenerator([out_poses_3d], [out_poses_2d], [out_actions]), batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers, pin_memory=True)

    prediction = evaluate(render_loader, model_pos, device, args.architecture)[0]

    # Invert camera transformation
    cam = dataset.cameras()[args.viz_subject][args.viz_camera]
    prediction = camera_to_world(prediction, R=cam['orientation'], t=0)
    prediction[:, :, 2] -= np.min(prediction[:, :, 2])
    ground_truth = camera_to_world(ground_truth, R=cam['orientation'], t=0)
    ground_truth[:, :, 2] -= np.min(ground_truth[:, :, 2])

    anim_output = {'Regression': prediction, 'Ground truth': ground_truth}
    input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])
    render_animation(input_keypoints, anim_output, dataset.skeleton(), dataset.fps(), args.viz_bitrate, cam['azimuth'],
                     args.viz_output, limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                     input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']),
                     input_video_skip=args.viz_skip)


def evaluate(data_loader, model_pos, device, architecture):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()
    epoch_loss_3d_pos_procrustes = AverageMeter()
    epoch_loss_3d_pos_scale = AverageMeter()

    predictions = []

    # Switch to evaluate mode
    torch.set_grad_enabled(False)
    model_pos.eval()
    end = time.time()


    bar = Bar('Eval ', max=len(data_loader))
    for i, (targets_3d, inputs_2d, _) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)  # targets_3d: (64, 16, 3)

        inputs_2d = inputs_2d.to(device)
        if architecture == 'linear':
            all_components = model_pos(inputs_2d.view(num_poses, -1))  # (64, 235)

            all_components = all_components.view(-1, 47, args.num_models)  # (64, 47, 5)
            outputs_3d_multiple = all_components[:, :45, :]  # (64, 45, 5)
            outputs_3d_multiple = outputs_3d_multiple.permute(0, 2, 1) # (64, 5, 45)
            outputs_3d_multiple = outputs_3d_multiple.view(-1, args.num_models, 15, 3).cpu()   # (64, 5, 15, 3)
            outputs_3d_all = torch.cat([torch.zeros(num_poses, args.num_models, 1, outputs_3d_multiple.size(3)), outputs_3d_multiple], 2) # Pad hip joint (0,0,0), (64, 5, 16, 3)

            # select the best 3D hypothesis
            dist_all = []
            for j in range(outputs_3d_all.shape[1]):
                dist_all.append(mpjpe(outputs_3d_all[:, j, :, :], targets_3d).item() * 1000.0)

            dist_all = torch.Tensor(dist_all)  # list to tensor
            dist = torch.min(dist_all, dim=0)
            index = dist[1]  # 2，第三个hypothesis
            #outputs_3d = outputs_3d_all[:, index, :, :]
            outputs_3d = outputs_3d_all[:, 4, :, :]

        # un-implemented
        else:
            outputs_3d = model_pos(inputs_2d).cpu()
            outputs_3d[:, :, :] -= outputs_3d[:, :1, :]  # Zero-centre the root (hip)

        predictions.append(outputs_3d.numpy())

        epoch_loss_3d_pos.update(mpjpe(outputs_3d, targets_3d).item() * 1000.0, num_poses)
        epoch_loss_3d_pos_procrustes.update(p_mpjpe(outputs_3d.numpy(), targets_3d.numpy()).item() * 1000.0, num_poses)
        epoch_loss_3d_pos_scale.update(n_mpjpe(outputs_3d, targets_3d).item() * 1000.0, num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}' \
            .format(batch=i + 1, size=len(data_loader), data=data_time.val, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, e1=epoch_loss_3d_pos.avg, e2=epoch_loss_3d_pos_procrustes.avg)
        bar.next()

    bar.finish()

    return np.concatenate(predictions), epoch_loss_3d_pos.avg, epoch_loss_3d_pos_procrustes.avg


if __name__ == '__main__':
    io, args = parse_args()
    main(args)