# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import torch.nn.functional as F


def weight_init(m):  # 参数初始化
    if isinstance(m, nn.Linear):  # 使用isinstance判断参数属于什么类型
        nn.init.kaiming_normal_(m.weight)  # 没有bias


class ResidualBlock(nn.Module):   # 基本的残差模块
    def __init__(self, linear_size, dropout_rate=0.5):
        super(ResidualBlock, self).__init__()
        self.l_size = linear_size

        self.linear1 = nn.Linear(self.l_size, self.l_size)
        self.bn1 = nn.BatchNorm1d(self.l_size)
        self.linear2 = nn.Linear(self.l_size, self.l_size)
        self.bn2 = nn.BatchNorm1d(self.l_size)

        self.relu = nn.ReLU(inplace=True)  # 节省空间
        self.dropout_rate = nn.Dropout(dropout_rate)

    def forward(self, x):
        y = self.linear1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.dropout_rate(y)

        y = self.linear2(y)
        y = self.bn2(y)
        y = self.relu(y)
        y = self.dropout_rate(y)

        out = x + y

        return out    # residual connection, point-wise addition


class LinearModel(nn.Module):
    def __init__(self, input_size, output_size, linear_size=1024,
                 num_stages=2, num_models=5, dropout_rate=0.5):
        super(LinearModel, self).__init__()

        self.linear_size = linear_size
        self.dropout_rate = dropout_rate
        self.num_stage = num_stages

        self.input_size = input_size #16*2
        self.output_size = output_size #16*3

        self.num_models = num_models  # specify the number of gaussian kernels in the mixture model

        self.linear1 = nn.Linear(self.input_size, self.linear_size)
        self.bn1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stage = []

        #for l in range(num_stages):
         #   self.linear_stage.append(ResidualBlock(self.linear_size, self.dropout_rate))
        #self.linear_stage = nn.ModuleList(self.linear_stage)

        self.linear_stage = nn.ModuleList(ResidualBlock(self.linear_size, self.dropout_rate) for i in range(num_stages))


        self.linear4 = nn.Linear(self.linear_size, self.output_size*self.num_models)
        self.linear5 = nn.Linear(self.linear_size, self.num_models)
        self.linear6 = nn.Linear(self.linear_size, self.num_models)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.elu = nn.ELU(inplace=True)


    def forward(self, x):

        y = self.linear1(x)  # x: (64, 32), y: (64, 1024)

        #print("linear.........{}".format(self.linear1.state_dict().items()))
        y = self.bn1(y)
        y = self.relu(y)
        y = self.dropout(y)

        #y = self.dropout(self.relu(self.linear1(x)))
        for j in range(self.num_stage):
            y = self.linear_stage[j](y)  # (64, 1024)
        y4 = self.linear4(y)  # (64, 45*5), remove root joint
        mu = y4


        y5 = self.linear5(y)
        sigma = self.elu(y5) + 1     # a modified elu function, shape: (64, 5)

        y6 = self.linear6(y)
        pi = F.softmax(y6, dim=1)   # (64, 5)
        components = torch.cat([mu, sigma, pi], dim=1)   # (64, 235)

        return components # (64, 45*5)



