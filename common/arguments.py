# -*- coding: utf-8 -*-

import argparse
import torch
import os
import errno
import datetime

from common.utils import IOStream

def parse_args():

    parser = argparse.ArgumentParser(description='Model training script')

    # experimental arguments
    parser.add_argument('--exp_name', default='Linear_MDN', type=str, choices=['SemGCN-GT', 'SemGCN-GT-NL', 'SemGCN-SH-NL'], metavar='N', help='Name of the experiment')
    parser.add_argument('--model', default='SemGCN', type=str, choices=['SemGCN'], metavar='N', help='Model to use')  # TODO(lz) Needs to implement
    parser.add_argument('--dataset', default='h36m', type=str, choices=['h36m', 'humanevas', 'MPI-INF-3DHP', 'motion-capture'], metavar='N', help='dataset to use')
    parser.add_argument('--architecture', default='linear', type=str, choices=[], help='network backbone architecture')
    parser.add_argument('--keypoints', default='sh_ft_h36m', type=str, choices=['gt', 'sh_ft_h36m'], metavar='N',
                        help='2D detections to use, [gt, sh_ft_h36m]')
    parser.add_argument('--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all actions')

    parser.add_argument("--evaluateActionWise", default=True, help="The dataset to use either h36m or heva")
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint/pretrained model to evaluate (filename)')
    parser.add_argument('--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (filename)')
    parser.add_argument('--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('--snapshot', default=10, type=int, help='save models for every snapshot epochs (default: 20)')
    parser.add_argument('--seed', default=1, type=int, help='random seed (default: 1)')


    # model arguments
    parser.add_argument('--num_blocks', default=4, type=int, metavar='N', help='number of residual blocks')
    parser.add_argument('--num_models', default=5, type=int, metavar='N', help='number of Gaussian models')
    parser.add_argument('--hidden_dim', default=128, type=int, metavar='N', help='number of hidden dimensions')
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR', help='downsample frame rate by factor')


    # train arguments
    parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='batch size in terms of predicted frames')
    parser.add_argument('--epochs', default=400, type=int, metavar='N', help='number of episode to train')
    parser.add_argument('--num_workers', default=8, type=int, metavar='N', help='number of workers for data loading')
    parser.add_argument('--no_max', dest='max_norm', action='store_false', help='if use max_norm clip on grid')
    parser.set_defaults(max_norm=True)

    parser.add_argument('--lr', default=1.0e-3, type=float, metavar='LR', help='initial learning rate (default: 0.001)')
    parser.add_argument('--lr_decay', default=100000, type=int, help='number of steps of learning rate decay')
    parser.add_argument('-lrd', '--lr-decay', default=0.95, type=float, metavar='LR',
                        help='learning rate decay per epoch')
    parser.add_argument('--lr_gamma', default=0.96, type=float, help='gamma of learning rate decay')
    parser.add_argument('--weight_decay', default=1e-4, type=int, metavar='WD', help='weight decay rate (default: 1e-4)')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate')
    parser.add_argument('--no_cuda', default=False, type=bool, help='enables CUDA training')

    # Visualization
    parser.add_argument('--viz_subject', type=str, metavar='STR', help='subject to render')
    parser.add_argument('--viz_action', type=str, metavar='STR', help='action to render')
    parser.add_argument('--viz_camera', type=int, default=0, metavar='N', help='camera to render')
    parser.add_argument('--viz_video', type=str, default=None, metavar='PATH', help='path to input video')
    parser.add_argument('--viz_skip', type=int, default=0, metavar='N', help='skip first N frames of input video')
    parser.add_argument('--viz_output', type=str, metavar='PATH', help='output file name (.gif or .mp4)')
    parser.add_argument('--viz_bitrate', type=int, default=3000, metavar='N', help='bitrate for mp4 videos')
    parser.add_argument('--viz_limit', type=int, default=-1, metavar='N', help='only render first N frames')
    parser.add_argument('--viz_downsample', type=int, default=1, metavar='N', help='downsample FPS by a factor N')
    parser.add_argument('--viz_size', type=int, default=5, metavar='N', help='image size')
    parser.add_argument("--train_dir", default="/home/lz/MDN_pytorch/experiments/test_git/", help="Data directory")

    args = parser.parse_args()


    # check invalid configuration
    if args.resume and args.evaluate:
        print('Invalid flags: --resume and --evaluate cannot be set at the same time')
        exit()


    args.cuda = not args.no_cuda and torch.cuda.is_available()


    if args.cuda:
        print('Using GPU: ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    else:
        print('Using CPU')

    try:
        CKPT_PATH = os.path.join('checkpoint', args.exp_name)
        os.makedirs(CKPT_PATH)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', CKPT_PATH)

    io = IOStream('checkpoint/' + args.exp_name + '/train.log')
    io.cprint(str(args))

    return io, args




