from __future__ import annotations
import torch
from torch.distributions.categorical import Categorical

def categorical_sample(logits: torch.Tensor, greedy: bool = False):
    if greedy:
        return torch.argmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    m = Categorical(probs)
    return m.sample()
