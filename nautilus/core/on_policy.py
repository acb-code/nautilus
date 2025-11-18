"""
Pure functional utilities for on-policy RL algorithms.

These functions implement:
- discounted returns (full-episode or reward-to-go)
- generalized advantage estimation (GAE)
- advantage normalization

All functions are numpy-only and free of side effects.
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. Returns / Discounted Reward Computations
# ---------------------------------------------------------------------------


def compute_returns_full_episode(
    rewards: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Compute *full-episode* discounted return R for every timestep:
        R_t = sum_{k=t..T} gamma^(k-t) * r_k
    But full-episode means we actually compute the *same* total return
    and assign it to each timestep.
    """
    T = len(rewards)
    indices = np.arange(T)
    discounts = np.power(gamma, indices)
    total_return = np.sum(rewards * discounts)
    return np.ones(T) * total_return


def compute_returns_reward_to_go(
    rewards: np.ndarray,
    gamma: float,
    bootstrap: float = 0.0,
) -> np.ndarray:
    """
    Compute reward-to-go returns:
        R_t = r_t + gamma r_{t+1} + ... + gamma^(n-t) * bootstrap

    Args:
        rewards: numpy array of episode rewards.
        gamma: discount factor.
        bootstrap: final value estimate V(s_T) or 0 if no value function.

    Returns:
        returns: array of reward-to-go values.
    """
    T = len(rewards)
    returns = np.zeros(T + 1)
    returns[-1] = bootstrap

    # Reverse-time accumulate
    for t in reversed(range(T)):
        returns[t] = rewards[t] + gamma * returns[t + 1]

    return returns[:-1]


# ---------------------------------------------------------------------------
# 2. GAE (Generalized Advantage Estimation)
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
    bootstrap_values: np.ndarray,
) -> np.ndarray:
    """
    Compute GAE over a *sequence of episodes* contained in a single rollout.

    Args:
        rewards: shape [T]
        values: shape [T]
        dones: boolean array shape [T] marking terminal timesteps
        gamma: discount factor
        lam: lambda for GAE
        bootstrap_values: array of value estimates at episode boundaries.
                          Typically one per episode.

    Notes:
        - We loop over episodes separated by dones.
        - For each episode, compute:
            delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            advantage_t = sum_{k=t..T} (gamma*lam)^(k-t) * delta_k

    Returns:
        advantages: numpy array shape [T]
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)

    # find episodes: indices where done=True
    terminals = np.where(dones.astype(int) == 1)[0]
    terminals = np.concatenate(([-1], terminals))  # include start boundary

    ep_idx = 0
    for ep_idx, (t0, t1) in enumerate(zip(terminals[:-1], terminals[1:], strict=False)):
        start = t0 + 1
        end = t1 + 1
        bootstrap_v = bootstrap_values[ep_idx]

        ep_rewards = rewards[start:end]
        ep_values = values[start:end]
        ep_next_values = np.concatenate((ep_values[1:], [bootstrap_v]))

        # delta_t = r_t + gamma V_{t+1} - V_t
        deltas = ep_rewards + gamma * ep_next_values - ep_values

        # compute discounted sum of deltas backwards
        gae = 0.0
        for t in reversed(range(len(deltas))):
            gae = deltas[t] + gamma * lam * gae
            advantages[start + t] = gae

    return advantages


# ---------------------------------------------------------------------------
# 3. Advantage normalization
# ---------------------------------------------------------------------------


def standardize_advantages(advantages: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Normalize advantages to have zero mean and unit variance.
    """
    mean = advantages.mean()
    std = advantages.std()
    return (advantages - mean) / (std + eps)
