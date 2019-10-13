
import torch
import torch.nn.functional as F
import numpy as np

def setup_seed(seed):

    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # 并行gpu
    torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致
    torch.backends.cudnn.benchmark = True  # 训练集变化不大时使训练加速




def mean_euclidean_distance(y_true, parameters,c,m ):
    """Mean Euclidean distance loss
    y_truth: ground truth 3d pose
    parameters: output of hypotheses generator, which conclude the mean, variance and mixture coefficient of the mixture model
    c: dimension of 3d pose
    m: number of kernels
    """
    components = parameters.view(-1, c + 2, m)  # c: 15*3, m: 5
    mu = components[:, :c, :]  # (64, 45, 5)
    dis = F.pairwise_distance((y_true.unsqueeze(2)), mu, p=2)   # (64, 5)
    print(dis.shape)
    res = torch.mean(dis)
    return res

setup_seed(1)
y_true = torch.randn(64, 45)

c = 45
m = 5

parameters = torch.randn(64, 47, 5)

res = mean_euclidean_distance(y_true, parameters, c, m)

print(res)
