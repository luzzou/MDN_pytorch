# -*- coding: utf-8 -*-
# pose_3d_loss + pose_length_loss


import os
import time
import datetime
import numpy as np

import torch
import torch.optim as optimizer
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from progress.bar import Bar
from common.data_utils import read_3d_data, create_2d_data, PoseGenerator
from common.arguments import parse_args
from common.log import Logger, savefig
from common.utils import AverageMeter, lr_decay, save_ckpt
from common.loss import mpjpe, p_mpjpe, n_mpjpe
from common.h36m_dataset import Human36mDataset, TRAIN_SUBJECTS, TEST_SUBJECTS
from models.mdn import LinearModel, weight_init
from common.camera import *
from common.train_loss import mean_log_Gaussian_like, Dirichlet_loss


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def fetch(subjects, dataset, keypoints, action_filter=None, stride=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_actions = []

    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    # if action.startswith(a):
                    if action.split(' ')[0] == a:
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])
                out_actions.append([action.split(' ')[0]] * poses_2d[i].shape[0])


            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_poses_3d) == 0:
        out_poses_3d = None

    if stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            out_actions[i] = out_actions[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_poses_3d, out_poses_2d, out_actions


def train(data_loader, model_pos, optimizer, device, lr_init, lr_current, step, decay, gamma, max_norm=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d = AverageMeter()
    epoch_loss_prior = AverageMeter()
    epoch_loss_total = AverageMeter()

    # Switch to train mode
    torch.set_grad_enabled(True)
    model_pos.train()
    end = time.time()

    bar = Bar('Train', max=len(data_loader))
    for i, (targets_3d, inputs_2d, _) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)   # 64, batch_size

        step += 1
        if step % decay == 0 or step == 1:
            lr_current = lr_decay(optimizer, step, lr_init, decay, gamma)

        # shape of targets_3d: (64, 15, 3), shape of input_2d: (64, 16, 2)
        targets_3d, inputs_2d = targets_3d[:, 1:, :].to(device), inputs_2d.to(device)  # Remove hip joint for 3D poses

        # prior也是一个超参数，需要在实验中慢慢调整
        #prior = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0]).to(device)
        prior = torch.tensor([2.0]).repeat(args.num_models).to(device)
        components = model_pos(inputs_2d.view(num_poses, -1))  # the shape of components: (64, 235)

        optimizer.zero_grad()

        # loss_prior只是对混合系数的约束，相当于正则项
        loss_prior = Dirichlet_loss(components, (num_joints-1) * 3, args.num_models, prior)

        # Mixture density network based on gaussian kernel
        targets_3d = targets_3d.view(num_poses, -1)  # (64, 45)

        loss_gaussion = mean_log_Gaussian_like(targets_3d, components, (num_joints-1)*3, args.num_models)
        loss_total = loss_prior + loss_gaussion
        ## 之前一直只反向传播了loss_gaussion，误差是61mm
        loss_total.backward()

        if max_norm:
            nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)
        optimizer.step()

        epoch_loss_total.update(loss_total.item(), num_poses)
        epoch_loss_3d.update(loss_gaussion.item(), num_poses)
        epoch_loss_prior.update(loss_prior.item(), num_poses)

        # Measure elapsed time
        batch_time.update((time.time() - end))
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data: .6f}s | Batch: {bt: .3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| total_loss: {total_loss: .4f} | 3D_loss: {pose_loss: .4f} | prior_loss: {prior_loss: .4f}'.format(batch=i + 1, size=len(data_loader), data=data_time.val,
                    bt=batch_time.avg,ttl=bar.elapsed_td, eta=bar.eta_td, total_loss=epoch_loss_total.avg, pose_loss=epoch_loss_3d.avg, prior_loss=epoch_loss_prior.avg)
        bar.next()

    bar.finish()
    return epoch_loss_total.avg, epoch_loss_3d.avg, epoch_loss_prior.avg, lr_current, step


def evaluate(data_loader, model_pos, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()
    epoch_loss_3d_pos_procrustes = AverageMeter()
    epoch_loss_3d_pos_scale = AverageMeter()

    # Switch to evaluate mode
    torch.set_grad_enabled(False)
    model_pos.eval()
    end = time.time()

    bar = Bar('Eval ', max=len(data_loader))
    for i, (targets_3d, inputs_2d, _) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)

        inputs_2d = inputs_2d.to(device)
        all_components = model_pos(inputs_2d.view(num_poses, -1))  # (64, 235)
        all_components = all_components.view(-1, (num_joints-1)*3+2, args.num_models)  # (64, 47, 5)
        outputs_3d_multiple = all_components[:, :(num_joints-1)*3, :]  # (64, 45, 5)
        outputs_3d_multiple = outputs_3d_multiple.permute(0, 2, 1) # (64, 5, 45)
        outputs_3d_multiple = outputs_3d_multiple.view(-1, args.num_models, num_joints-1, 3).cpu()   # (64, 5, 15, 3)
        outputs_3d_all = torch.cat([torch.zeros(num_poses, args.num_models, 1, outputs_3d_multiple.size(3)), outputs_3d_multiple], 2) # Pad hip joint (0,0,0), (64, 5, 16, 3)

        # select the best 3D hypothesis
        dist_all = []
        for j in range(outputs_3d_all.shape[1]):
            dist_all.append(mpjpe(outputs_3d_all[:, j, :, :], targets_3d).item() * 1000.0)

        dist_all = torch.Tensor(dist_all)  # list to tensor
        dist = torch.min(dist_all, dim=0)
        index = dist[1]
        outputs_3d = outputs_3d_all[:, index, :, :]

        epoch_loss_3d_pos.update(mpjpe(outputs_3d, targets_3d).item() * 1000.0, num_poses)
        epoch_loss_3d_pos_procrustes.update(p_mpjpe(outputs_3d.numpy(), targets_3d.numpy()).item() * 1000.0, num_poses)
        epoch_loss_3d_pos_scale.update(n_mpjpe(outputs_3d, targets_3d).item() * 1000.0, num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f} | N-MPJPE: {e3: .4f}' \
                     .format(batch=i + 1, size=len(data_loader), data=data_time.val, bt=batch_time.avg,
                             ttl=bar.elapsed_td, eta=bar.eta_td, e1=epoch_loss_3d_pos.avg,
                             e2=epoch_loss_3d_pos_procrustes.avg, e3=epoch_loss_3d_pos_scale.avg)
        bar.next()

    bar.finish()
    return epoch_loss_3d_pos.avg, epoch_loss_3d_pos_procrustes.avg, epoch_loss_3d_pos_scale.avg



if __name__ == '__main__':

    io, args = parse_args()
    io.cprint('==> loading 3D joint dataset...')
    dataset_path = os.path.join('data', 'data_3d_' + args.dataset + '.npz')  # .npz数据集的介绍
    dataset = Human36mDataset(dataset_path) # 删除静态点，调整关节点索引关系,合并内、外参矩阵并进行归一化，修正错误的父节点，将相机参数加入到数据集中,
                                            # subject/action有两个键dict_keys(['positions', 'cameras'])
    subjects_train = TRAIN_SUBJECTS
    subjects_test = TEST_SUBJECTS

    dataset = read_3d_data(dataset)  # 通过相机外参将joints从世界坐标系变换到相机坐标系，并消除全局偏移(以第一个节点Pelvis作为根节点), 将变换后的数据新增加到数据集中，新增的键为'positions_3d'
                                     # dataset: /subject/action: dict_keys(['positions'], ['cameras'], 'positions_3d')
    keypoints_path = os.path.join('data', 'data_2d_' + args.dataset + '_' + args.keypoints + '.npz')
    keypoints = create_2d_data(keypoints_path, dataset) # 将2D pose归一化到[-1, 1], 2D数据集中只有['positions_2d'],需要使用3D数据集中的['cameras']中保存的图像分辨率进行归一化

    action_filter = None if args.actions == '*' else args.actions.split(',')
    if action_filter is not None:
        action_filter = map(lambda x: dataset.define_actions(x)[0], action_filter)  # map有两个参数，一个函数，一个可迭代序列，将序列中的每一个元素作为函数参数进行运算加工
        io.cprint('==> Selected actions: {}'.format(action_filter))  # e.g. ==> Selected actions: ['Directions', 'Discussion']

    stride = args.downsample
    cudnn.benchmark = True  # 加快运行效率，仅限于输入数据变化不大的情况
    device = torch.device('cuda' if args.cuda else 'cpu')

    num_joints = dataset.skeleton().num_joints()  # 16

    # Create model
    model_pos = LinearModel(num_joints * 2, (num_joints - 1) * 3, num_models=args.num_models).to(device)
    model_pos.apply(weight_init)

    # io.cprint(str(model_pos))
    io.cprint('==> Total parameters: {:.2f}M'.format(sum(p.numel() for p in model_pos.parameters()) / 1000000.0))
    io.cprint('Use Adam')
    optim = optimizer.Adam(model_pos.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optim = optim.Adam(model_pos.parameters(), lr=args.lr)
    # weight decay的作用是调节模型复杂度对损失函数的影响,防止过拟合

    # Optionally resume from a checkpoint
    if args.resume or args.evaluate:
        ckpt_path = (args.resume if args.resume else args.evaluate)

        if os.path.isfile(ckpt_path):
            io.cprint("==> Loading checkpoint '{}'".format(ckpt_path))
            ckpt = torch.load(ckpt_path)
            start_epoch = ckpt['epoch']
            error_best = ckpt['error']
            glob_step = ckpt['step']
            lr_current = ckpt['lr']
            model_pos.load_state_dict(ckpt['state_dict'])
            optim.load_state_dict(ckpt['optimizer'])
            io.cprint("==> Loaded checkpoint (Epoch: {} | Error: {})".format(start_epoch, error_best))

            if args.resume:
                ckpt_dir_path = os.path.dirname(ckpt_path)
                logger = Logger(os.path.join(ckpt_dir_path, 'result.log'), resume=True)

        else:
            raise RuntimeError("==> No checkpoint found at '{}'".format(ckpt_path))

    # Prepare for training from scratch
    else:
        start_epoch = 0
        error_best = None
        glob_step = 0
        lr_current = args.lr
        ckpt_dir_path = os.path.join(args.checkpoint, args.exp_name, datetime.datetime.now().isoformat())

        if not os.path.exists(ckpt_dir_path):
            os.makedirs(ckpt_dir_path)
            io.cprint('==> Making checkpoint dir: {}'.format(ckpt_dir_path))

        logger = Logger(os.path.join(ckpt_dir_path, 'result.log'))
        logger.set_names(['epoch', 'lr', 'loss_train', 'loss_train_3d', 'loss_train_prior', 'error_eval_p1', 'error_eval_p2', 'error_eval_p3'])

    # Testing the well-trained model
    if args.evaluate:
        io.cprint("==> Evaluating...")

        if action_filter is None:
            action_filter = dataset.define_actions()

        errors_p1 = np.zeros(len(action_filter))
        errors_p2 = np.zeros(len(action_filter))
        errors_p3 = np.zeros(len(action_filter))

        model_pos = nn.DataParallel(model_pos)
        cudnn.benchmark = True
        for i, action in enumerate(action_filter):
            poses_valid_3d, poses_valid_2d, actions_valid = fetch(subjects_test, dataset, keypoints, [action], stride)
            valid_loader = DataLoader(PoseGenerator(poses_valid_3d, poses_valid_2d, actions_valid),
                                          batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.num_workers, pin_memory=True)
            errors_p1[i], errors_p2[i], errors_p3[i] = evaluate(valid_loader, model_pos, device)

        io.cprint('Protocol #1 (MPJPE) action-wise average: {:.2f}(mm)'.format(np.mean(errors_p1).item()))
        io.cprint('Protocol #2 (P-MPJPE) action-wise average: {:.2f}(mm)'.format(np.mean(errors_p2).item()))
        io.cprint('Protocol #3 (N-MPJPE) action-wise average: {:.2f}(mm)'.format(np.mean(errors_p3).item()))
        exit(0)


    # Start Model Training
    poses_train_3d, poses_train_2d, actions_train = fetch(subjects_train, dataset, keypoints, action_filter, stride)
    train_loader = DataLoader(PoseGenerator(poses_train_3d, poses_train_2d, actions_train),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)

    poses_valid_3d, poses_valid_2d, actions_valid = fetch(subjects_test, dataset, keypoints, action_filter, stride)
    valid_loader = DataLoader(PoseGenerator(poses_valid_3d, poses_valid_2d, actions_valid),
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: %d | LR: %.8f' % ((epoch + 1), lr_current))

        # Train for one epoch
        epoch_loss_total, epoch_loss_3d, epoch_loss_prior, lr_current, glob_step = train(train_loader, model_pos, optim, device, args.lr,
                                                  lr_current,
                                                  glob_step, args.lr_decay, args.lr_gamma, max_norm=args.max_norm)

        #  Evaluate during training
        error_eval_p1, error_eval_p2, error_eval_p3 = evaluate(valid_loader, model_pos, device)

        # Update log file
        logger.append([epoch + 1, lr_current, epoch_loss_total, epoch_loss_3d, epoch_loss_prior, error_eval_p1, error_eval_p2, error_eval_p3])

        # Save checkpoint   ==> 保留最好的模型
        if error_best is None or error_best > error_eval_p1:
             error_best = error_eval_p1
             save_ckpt({'epoch': epoch + 1, 'lr': lr_current, 'step': glob_step, 'state_dict': model_pos.state_dict(),
                        'optimizer': optim.state_dict(), 'error': error_eval_p1}, ckpt_dir_path, suffix='best')

        # 训练一定epochs之后保留一次模型
        if (epoch + 1) % args.snapshot == 0:
            save_ckpt({'epoch': epoch + 1, 'lr': lr_current, 'step': glob_step, 'state_dict': model_pos.state_dict(),
                       'optimizer': optim.state_dict()}, ckpt_dir_path)

    logger.close()

    logger.plot(['loss_train', 'loss_train_3d', 'loss_train_prior', 'error_eval_p1', 'error_eval_p2', 'error_eval_p3'])
    savefig(os.path.join(ckpt_dir_path, 'log.eps'))


