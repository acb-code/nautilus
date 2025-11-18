"""
Utilities for on-policy RL algorithms.

Includes:
- a lightweight rollout buffer for on-policy methods
- discounted returns (full-episode or reward-to-go)
- generalized advantage estimation (GAE)
- advantage normalization
"""

import numpy as np

# ---------------------------------------------------------------------------
# 0. Rollout Buffer for On-Policy Algorithms
# ---------------------------------------------------------------------------


class OnPolicyBuffer:
    """
    Lightweight rollout storage for on-policy algorithms like PPO/VPG.

    Stores per-timestep data for vectorized environments and exposes helpers
    to compute GAE/returns.
    """

    def __init__(self, rollout_length: int, num_envs: int):
        self.rollout_length = rollout_length
        self.num_envs = num_envs
        self.reset()

    def reset(self):
        self.obs: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.rewards: list[np.ndarray] = []
        self.dones: list[np.ndarray] = []
        self.infos: list[dict] = []
        self.values: list[np.ndarray] = []
        self.log_probs: list[np.ndarray] = []
        self.advantages: np.ndarray | None = None
        self.returns: np.ndarray | None = None

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        info: dict | None = None,
    ):
        """
        Append one rollout step. `info` may contain 'val' and 'log_prob'.
        """
        self.obs.append(np.asarray(obs))
        self.actions.append(np.asarray(action))
        self.rewards.append(np.asarray(reward))
        self.dones.append(np.asarray(done))

        if info is None:
            info = {}
        self.infos.append(info)

        if "val" in info:
            self.values.append(np.asarray(info["val"]))
        if "log_prob" in info:
            self.log_probs.append(np.asarray(info["log_prob"]))

    def compute_returns_and_advantages(
        self,
        last_value: np.ndarray,
        gamma: float,
        lam: float,
        last_done: np.ndarray | None = None,
        normalize_advantages: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute (and cache) GAE advantages and returns.
        """
        if not self.values:
            raise ValueError("Cannot compute advantages without stored values.")

        rewards = np.asarray(self.rewards)
        values = np.asarray(self.values)
        dones = np.asarray(self.dones, dtype=np.float32)
        next_done = np.zeros_like(last_value) if last_done is None else np.asarray(last_done)

        advantages = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=gamma,
            lam=lam,
            next_value=np.asarray(last_value),
            next_done=next_done,
        )
        returns = advantages + values

        if normalize_advantages:
            flat_adv = standardize_advantages(advantages.flatten())
            advantages = flat_adv.reshape(advantages.shape)

        self.advantages = advantages
        self.returns = returns
        return advantages, returns

    def get(self) -> dict[str, np.ndarray]:
        """
        Return the rollout as stacked numpy arrays.
        """
        batch = {
            "obs": np.asarray(self.obs),
            "actions": np.asarray(self.actions),
            "rewards": np.asarray(self.rewards),
            "dones": np.asarray(self.dones),
            "infos": self.infos,
        }

        if self.values:
            batch["values"] = np.asarray(self.values)
        if self.log_probs:
            batch["log_probs"] = np.asarray(self.log_probs)
        if self.advantages is not None:
            batch["advantages"] = np.asarray(self.advantages)
        if self.returns is not None:
            batch["returns"] = np.asarray(self.returns)

        return batch


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
    next_value: np.ndarray,
    next_done: np.ndarray,
) -> np.ndarray:
    """
    Compute Generalized Advantage Estimation (GAE) for a rollout segment.

    Args:
        rewards: Shape (T, N)
        values: Shape (T, N) - V(s_t)
        dones: Shape (T, N) - 1.0 if s_t was terminal, 0.0 otherwise
        gamma: Discount factor
        lam: GAE lambda
        next_value: Shape (N,) - V(s_{T+1}), used for bootstrapping
        next_done: Shape (N,) - 1.0 if s_{T+1} is terminal (usually 0 for timeouts)

    Returns:
        advantages: Shape (T, N)
    """
    advantages = np.zeros_like(rewards)
    last_gae_lam = 0

    # Iterate backwards from T-1 to 0
    num_steps = rewards.shape[0]

    for t in reversed(range(num_steps)):
        # If the trajectory ended at timestep t, we should not bootstrap further.
        next_non_terminal = 1.0 - (next_done if t == num_steps - 1 else dones[t])
        next_val = next_value if t == num_steps - 1 else values[t + 1]

        # Delta = r_t + gamma * V(s_{t+1}) * (1-d_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]

        # GAE = delta + gamma * lambda * (1-d_{t+1}) * GAE_{t+1}
        last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
        advantages[t] = last_gae_lam

    return advantages


def standardize_advantages(advantages: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize advantages (mean 0, std 1).
    """
    return (advantages - np.mean(advantages)) / (np.std(advantages) + eps)
