
import torch

d=torch.Tensor([[1,3],[2,4]])
print(torch.max(d,0))
print(torch.max(d,1))