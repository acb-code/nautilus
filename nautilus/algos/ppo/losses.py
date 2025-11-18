"""
PPO / on-policy loss functions.

All losses operate on torch tensors and do not depend on any class.
This makes them reusable by any PPO-style trainer.
"""

import jax.numpy as jnp
import numpy as np  # for np.pi and np.e
import torch
import torch.nn.functional as F

# Define PI and E for JAX equivalent of torch.pi and torch.e
JAX_PI = np.pi
JAX_E = np.e

# ---------------------------------------------------------------------------
# 1. Value loss
# ---------------------------------------------------------------------------


def value_loss(
    values: torch.Tensor,
    target_values: torch.Tensor,
    clip: float = None,
):
    """
    Standard MSE value loss, or clipped value loss if `clip` is provided.

    Args:
        values: predicted V(s)
        target_values: returns computed from GAE or reward-to-go
        clip: optional PPO-style value clipping

    Returns:
        value_loss: scalar tensor
    """
    if clip is None or clip < 0:
        return F.mse_loss(values, target_values)

    # Clipped value loss (PPO2 style)
    values_clipped = torch.clamp(
        values,
        target_values - clip,
        target_values + clip,
    )
    loss1 = (values - target_values) ** 2
    loss2 = (values_clipped - target_values) ** 2
    return torch.mean(torch.max(loss1, loss2))


# ---------------------------------------------------------------------------
# 2. Entropy bonuses
# ---------------------------------------------------------------------------


def entropy_discrete(policy_probs: torch.Tensor, eps: float = 1e-8):
    """
    Entropy for a categorical distribution.
    """
    policy_probs = torch.clamp(policy_probs, eps, 1.0)
    return -torch.sum(policy_probs * torch.log(policy_probs), dim=-1).mean()


def entropy_gaussian(std: torch.Tensor):
    """
    Entropy of a factorized Gaussian distribution with diagonal std.
    """
    return torch.mean(torch.log(std) + 0.5 * torch.log(2 * torch.pi * torch.e))


# ---------------------------------------------------------------------------
# 3. KL divergence
# ---------------------------------------------------------------------------


def kl_discrete(old_probs: torch.Tensor, new_probs: torch.Tensor, eps: float = 1e-8):
    old = torch.clamp(old_probs, eps, 1.0)
    new = torch.clamp(new_probs, eps, 1.0)
    return torch.sum(old * (torch.log(old) - torch.log(new)), dim=-1).mean()


def kl_gaussian(
    mu_old: torch.Tensor,
    std_old: torch.Tensor,
    mu_new: torch.Tensor,
    std_new: torch.Tensor,
    eps: float = 1e-8,
):
    """
    KL divergence between diagonal Gaussians.
    """
    var_old = std_old.pow(2)
    var_new = std_new.pow(2)

    return torch.mean(
        torch.sum(
            torch.log(std_new / (std_old + eps))
            + (var_old + (mu_old - mu_new).pow(2)) / (2 * var_new + eps)
            - 0.5,
            dim=-1,
        )
    )


def approximate_kl(old_log_probs, new_log_probs):
    """Spinning Up style approximate KL."""
    return (old_log_probs - new_log_probs).mean()


# ---------------------------------------------------------------------------
# 4. PPO policy loss
# ---------------------------------------------------------------------------


def ppo_policy_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float,
):
    """
    PPO clipped surrogate objective:
        L = min( r * A, clip(r, 1-eps, 1+eps) * A )
    """
    ratio = torch.exp(new_log_probs - old_log_probs)

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages

    loss = -torch.mean(torch.min(unclipped, clipped))
    return loss


# ---------------------------------------------------------------------------
# 1. Value loss (JAX)
# ---------------------------------------------------------------------------


def value_loss_jax(
    values: jnp.ndarray,
    target_values: jnp.ndarray,
    clip: float = None,
):
    """
    Standard MSE value loss, or clipped value loss if `clip` is provided.
    (JAX implementation)
    """
    if clip is None or clip < 0:
        # Note: JAX doesn't have a direct MSE loss in jnp,
        # so we implement it as the mean of squared differences.
        return jnp.mean((values - target_values) ** 2)

    # Clipped value loss (PPO2 style)
    values_clipped = jnp.clip(
        values,
        target_values - clip,
        target_values + clip,
    )
    loss1 = (values - target_values) ** 2
    loss2 = (values_clipped - target_values) ** 2
    # jnp.maximum is the JAX equivalent of torch.max for element-wise comparison
    return jnp.mean(jnp.maximum(loss1, loss2))


# ---------------------------------------------------------------------------
# 2. Entropy bonuses (JAX)
# ---------------------------------------------------------------------------


def entropy_discrete_jax(policy_probs: jnp.ndarray, eps: float = 1e-8):
    """
    Entropy for a categorical distribution (JAX implementation).
    """
    # jnp.clip is the JAX equivalent of torch.clamp
    policy_probs = jnp.clip(policy_probs, eps, 1.0)
    # jnp.sum and jnp.log are JAX equivalents
    return -jnp.sum(policy_probs * jnp.log(policy_probs), axis=-1).mean()


def entropy_gaussian_jax(std: jnp.ndarray):
    """
    Entropy of a factorized Gaussian distribution with diagonal std. (JAX implementation)
    """
    # Note: Using numpy's PI and E constants, as JAX's jnp doesn't expose them directly
    # and they are just constants for the calculation.
    return jnp.mean(jnp.log(std) + 0.5 * jnp.log(2 * JAX_PI * JAX_E))


# ---------------------------------------------------------------------------
# 3. KL divergence (JAX)
# ---------------------------------------------------------------------------


def kl_discrete_jax(old_probs: jnp.ndarray, new_probs: jnp.ndarray, eps: float = 1e-8):
    # jnp.clip is the JAX equivalent of torch.clamp
    old = jnp.clip(old_probs, eps, 1.0)
    new = jnp.clip(new_probs, eps, 1.0)
    # jnp.sum and jnp.log are JAX equivalents
    return jnp.sum(old * (jnp.log(old) - jnp.log(new)), axis=-1).mean()


def kl_gaussian_jax(
    mu_old: jnp.ndarray,
    std_old: jnp.ndarray,
    mu_new: jnp.ndarray,
    std_new: jnp.ndarray,
    eps: float = 1e-8,
):
    """
    KL divergence between diagonal Gaussians (JAX implementation).
    """
    # jnp.power and standard operators are used
    var_old = jnp.power(std_old, 2)
    var_new = jnp.power(std_new, 2)

    return jnp.mean(
        jnp.sum(
            jnp.log(std_new / (std_old + eps))
            + (var_old + jnp.power((mu_old - mu_new), 2)) / (2 * var_new + eps)
            - 0.5,
            axis=-1,
        )
    )


def approximate_kl_jax(old_log_probs, new_log_probs):
    """Spinning Up style approximate KL (JAX implementation)."""
    return (old_log_probs - new_log_probs).mean()


# ---------------------------------------------------------------------------
# 4. PPO policy loss (JAX)
# ---------------------------------------------------------------------------


def ppo_policy_loss_jax(
    new_log_probs: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    advantages: jnp.ndarray,
    clip_ratio: float,
):
    """
    PPO clipped surrogate objective (JAX implementation).
    """
    # jnp.exp is the JAX equivalent of torch.exp
    ratio = jnp.exp(new_log_probs - old_log_probs)

    unclipped = ratio * advantages
    # jnp.clip is the JAX equivalent of torch.clamp
    clipped = jnp.clip(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages

    # jnp.minimum is the JAX equivalent of torch.min for element-wise comparison
    loss = -jnp.mean(jnp.minimum(unclipped, clipped))
    return loss
