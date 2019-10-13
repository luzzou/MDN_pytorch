

import torch
import torch.nn as nn
import os
import numpy as np
import numpy.random as random
from common.loss import mpjpe, p_mpjpe, n_mpjpe

def seed_torch(seed=2018):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch()

outputs_3d_multiple = torch.randn(64, 45, 5)
outputs_3d_multiple = outputs_3d_multiple.permute(0, 2, 1)  # (64, 5, 45)
outputs_3d_multiple = outputs_3d_multiple.view(-1, 5, 15, 3)   # (64, 5, 15, 3)

target = torch.randn(64, 16, 3)

outputs_3d_pad = torch.cat([torch.zeros(64, 5, 1, 3), outputs_3d_multiple], 2)  #(64, 5, 16, 3)


criterion = nn.MSELoss(reduction='mean')

dist = []
a = []

#print(dist)
    # epoch_loss_3d_pos.update(mpjpe(outputs_3d, targets_3d).item() * 1000.0, num_poses)
    # epoch_loss_3d_pos_procrustes.update(p_mpjpe(outputs_3d.numpy(), targets_3d.numpy()).item() * 1000.0, num_poses)
    # epoch_loss_3d_pos_scale.update(n_mpjpe(outputs_3d, targets_3d).item() * 1000.0, num_poses)



# print(outputs_3d_multiple.shape)   # (64, 16, 3, 5)
#
# outputs_3d = torch.cat([torch.zeros(num_poses, 1, outputs_3d.size(2)), outputs_3d], 1)  # Pad hip joint (0,0,0)


   # (batch_size, 16,3)

for i in range(outputs_3d_pad.shape[1]):
    #print(outputs_3d_pad[:, i, :, :].shape)  # (64, 16, 3)
    dist.append(mpjpe(outputs_3d_pad[:, i, :, :], target).item() * 1000.0)

dist_all = torch.Tensor(dist)
dist = torch.min(dist_all, dim=0)
index = dist[1]  # 选择最小的MPJPE对应的索引
outputs_3d = outputs_3d_pad[:, index, :, :]

print(outputs_3d.shape)

# outputs_3d_multiple = outputs_3d_multiple.cpu()
#         outputs_3d_multiple = outputs_3d_multiple.permute(0, 2, 1)  # (64, 5, 45)
#         outputs_3d_multiple = outputs_3d_multiple.view(-1, args.num_models, num_joints-1, 3)   # (64, 5, 15, 3)
#         outputs_3d_all = torch.cat([torch.zeros(num_poses, args.num_models, 1, outputs_3d_multiple.size(3)), outputs_3d_multiple], 2) # Pad hip joint (0,0,0), (64, 5, 16, 3)
#
#         # select the best 3D hypothesis
#         dist_all = []
#         for i in range(outputs_3d_all):
#             dist_all.append(mpjpe(outputs_3d_all[:, i, :, :], targets_3d).item() * 1000.0)
#         dist_all = torch.Tensor(dist_all)
#         dist = torch.min(dist_all)
#         index = dist[1]  # 选择最小的MPJPE对应的索引
#         outputs_3d = outputs_3d_all[:, index, :, :]
#
#         epoch_loss_3d_pos.update(mpjpe(outputs_3d, targets_3d).item() * 1000.0, num_poses)
#         epoch_loss_3d_pos_procrustes.update(p_mpjpe(outputs_3d.numpy(), targets_3d.numpy()).item() * 1000.0, num_poses)
#         epoch_loss_3d_pos_scale.update(n_mpjpe(outputs_3d, targets_3d).item() * 1000.0, num_poses)

