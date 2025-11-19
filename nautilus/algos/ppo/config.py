from dataclasses import dataclass

from nautilus.core.policy_optimizer_base import OptimizerConfig


@dataclass
class PPOConfig(OptimizerConfig):
    # Hyperparameters
    gamma: float = 0.99
    lam: float = 0.95  # GAE lambda
    clip_ratio: float = 0.2  # PPO clip ratio
    target_kl: float | None = None  # Target KL for early stopping (CleanRL default: None)
    lr_decay: bool = False  # Linearly decay learning rates to zero over training
    minibatch_size: int | None = None  # SGD minibatch size (full batch if None)
    update_epochs: int | None = 4  # Number of passes over each batch (CleanRL default: 4)
    entropy_coef: float = 0.01  # Entropy bonus weight
    max_grad_norm: float = 0.5  # Gradient clipping threshold
    vf_coef: float = 0.5  # Value loss coefficient (CleanRL default)
    clip_vloss: bool = True  # PPO2-style value clipping
    norm_adv: bool = True  # Normalize advantages (CleanRL default)

    # Optimization
    pi_lr: float = 3e-4
    vf_lr: float = 1e-3
    train_pi_iter: int = 4  # Policy update steps per batch (fallback if update_epochs is None)
    train_v_iter: int = 4  # Value update steps per batch (fallback if update_epochs is None)

    # Architecture / Setup
    actor_hidden_sizes: tuple = (64, 64)
    critic_hidden_sizes: tuple = (64, 64)
    normalize: bool = False
