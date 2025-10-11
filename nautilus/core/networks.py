from __future__ import annotations
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden=(128,128), activation='relu'):
        super().__init__()
        acts = {'relu': nn.ReLU, 'tanh': nn.Tanh}
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), acts[activation]()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
