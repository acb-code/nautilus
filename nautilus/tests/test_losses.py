import numpy as np
import torch

from nautilus.algos.ppo.losses import (
    approximate_kl,
    entropy_discrete,
    entropy_gaussian,
    ppo_policy_loss,
    value_loss,
)


def test_value_loss_clipping_matches_max_squared_error():
    values = torch.tensor([0.0, 2.0])
    targets = torch.tensor([1.0, 0.0])

    loss = value_loss(values, targets, clip=0.5)

    # Manual calculation:
    # loss1 = [(0-1)^2, (2-0)^2] = [1, 4]
    # loss2 = [(0.5-1)^2, (0.5-0)^2] = [0.25, 0.25]
    # max -> [1, 4], mean -> 2.5
    assert torch.isclose(loss, torch.tensor(2.5))


def test_entropy_helpers():
    probs = torch.tensor([[0.5, 0.5]])
    ent = entropy_discrete(probs)
    expected = torch.tensor(np.log(2.0), dtype=ent.dtype, device=ent.device)
    assert torch.isclose(ent, expected, atol=1e-6)

    std = torch.ones(2)
    ent_gauss = entropy_gaussian(std)
    constant = torch.tensor(2 * torch.pi * torch.e, dtype=std.dtype, device=std.device)
    expected = torch.log(std) + 0.5 * torch.log(constant)
    assert torch.isclose(ent_gauss, expected.mean())


def test_ppo_policy_loss_applies_clipping_symmetrically():
    old_log_probs = torch.log(torch.tensor([0.5, 0.5]))
    new_log_probs = torch.log(torch.tensor([0.6, 0.4]))
    advantages = torch.tensor([1.0, -1.0])

    loss = ppo_policy_loss(new_log_probs, old_log_probs, advantages, clip_ratio=0.1)

    # Ratios: [1.2, 0.8] -> clipped: [1.1, 0.9]
    # Surrogate: [min(1.2,1.1)*1, min(0.8,0.9)*-1] = [1.1, -0.9]
    expected = -torch.mean(torch.tensor([1.1, -0.9]))
    assert torch.isclose(loss, expected)


def test_approximate_kl_matches_manual_mean():
    old = torch.tensor([-0.7, -0.7])
    new = torch.tensor([-0.5, -1.0])

    kl = approximate_kl(old, new)

    manual = torch.mean(old - new)
    assert torch.isclose(kl, manual)
