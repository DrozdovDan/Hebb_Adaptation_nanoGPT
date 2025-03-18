import torch 
from torch import nn

x = torch.randn(10)

a = nn.Linear(10, 5)

nn.init.normal_(a.weight)

b = nn.Linear(5, 10)

nn.init.orthogonal_(b.weight)

print(torch.norm(x))

x = a(x)

print(torch.norm(x))

x = b(x)

print(torch.norm(x))