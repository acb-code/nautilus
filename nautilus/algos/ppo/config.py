from dataclasses import dataclass

from nautilus.core.policy_optimizer_base import OptimizerConfig


@dataclass
class PPOConfig(OptimizerConfig):
    # Hyperparameters
    gamma: float = 0.99
    lam: float = 0.95  # GAE lambda
    clip_ratio: float = 0.2  # PPO clip ratio
    target_kl: float = 0.01  # Target KL for early stopping

    # Optimization
    pi_lr: float = 3e-4
    vf_lr: float = 1e-3
    train_pi_iter: int = 80  # Policy update steps per batch
    train_v_iter: int = 80  # Value update steps per batch

    # Architecture / Setup
    actor_hidden_sizes: tuple = (64, 64)
    critic_hidden_sizes: tuple = (64, 64)
