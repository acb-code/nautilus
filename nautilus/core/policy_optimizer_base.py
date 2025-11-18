"""
A lightweight base class for on-policy RL training.

This class handles:
- config
- environment creation
- rollout collection
- standard training loop
- logging hook points
- checkpointing hooks

It intentionally does NOT implement algorithm-specific logic.
Algorithms should implement:
    - select_action()
    - compute_losses()
    - update_params()
"""

import contextlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from nautilus.envs.api_compat import reset_env, step_env


@dataclass
class OptimizerConfig:
    seed: int = 0
    total_steps: int = 1_000_000
    rollout_length: int = 2048
    max_ep_length: int = 1000
    device: str = "cpu"
    backend: str = "torch"
    log_interval: int = 10_000
    save_interval: int = 50_000
    save_path: str = "./checkpoints"


class PolicyOptimizerBase:
    """
    Base class providing:
    - setup
    - training loop
    - rollout gathering
    - hooks algorithms override
    """

    def __init__(self, env_fn, config: OptimizerConfig):
        self.env_fn = env_fn
        self.config = config
        self._setup_backend()

        # device
        self.device = torch.device(config.device)

        # environment
        self.env = env_fn()
        self.obs, _ = reset_env(self.env)

        # rng
        self._seed_everything(config.seed)

        # step counter
        self.total_steps = 0

        # storage for logs
        self.ep_returns = []
        self.ep_lengths = []
        self.current_return = 0
        self.current_length = 0

    # ------------------------------------------------------------------
    # Setup utilities
    # ------------------------------------------------------------------

    def _seed_everything(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)

        with contextlib.suppress(TypeError):
            self.env.reset(seed=seed)

    # ------------------------------------------------------------------
    # Algorithm-specific methods — to be overridden
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray) -> tuple[Any, dict]:
        """
        Algorithm implements:
        - run policy network
        - sample action
        - compute log_prob
        - return (action, info_dict)
        """
        raise NotImplementedError

    def compute_losses(self, batch: dict) -> dict[str, torch.Tensor]:
        """
        Algorithm implements:
        - compute policy loss
        - compute value loss
        - compute entropy bonus
        Return dict of losses.
        """
        raise NotImplementedError

    def update_params(self, losses: dict[str, torch.Tensor]):
        """
        Algorithm implements:
        - call optimizer.step()
        - update network params
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Rollout Collection
    # ------------------------------------------------------------------

    def collect_rollout(self) -> dict[str, np.ndarray]:
        """
        Collect a fixed-length rollout from the policy.
        Returns a dict containing:
            obs, actions, rewards, dones, infos
        """
        obs_buf = []
        actions_buf = []
        rewards_buf = []
        dones_buf = []
        infos_buf = []

        for _t in range(self.config.rollout_length):
            # select action from policy
            action, info = self.select_action(self.obs)

            # environment step
            next_obs, reward, terminated, truncated, env_info = step_env(
                self.env, action, truncated=False
            )

            done = terminated or truncated

            # buffers
            obs_buf.append(self.obs)
            actions_buf.append(action)
            rewards_buf.append(reward)
            dones_buf.append(done)
            infos_buf.append(info)

            # episode stats
            self.current_return += reward
            self.current_length += 1

            if done or self.current_length >= self.config.max_ep_length:
                self.ep_returns.append(self.current_return)
                self.ep_lengths.append(self.current_length)

                self.current_return = 0
                self.current_length = 0

                next_obs, _ = reset_env(self.env)

            self.obs = next_obs
            self.total_steps += 1

        return dict(
            obs=np.array(obs_buf),
            actions=np.array(actions_buf),
            rewards=np.array(rewards_buf),
            dones=np.array(dones_buf),
            infos=infos_buf,
        )

    # ------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------

    def train(self):
        """
        Standard on-policy training loop:
        - collect rollout
        - compute advantages & returns
        - compute losses
        - update parameters
        """

        while self.total_steps < self.config.total_steps:
            batch = self.collect_rollout()

            # Let the algorithm compute returns, GAE, losses
            losses = self.compute_losses(batch)

            # Algorithm updates model
            self.update_params(losses)

            # Logging
            if self.total_steps % self.config.log_interval < self.config.rollout_length:
                self.log_status(losses)

            # Saving
            if self.total_steps % self.config.save_interval < self.config.rollout_length:
                self.save_checkpoint()

    # ------------------------------------------------------------------
    # Hooks for logging & saving
    # ------------------------------------------------------------------

    def log_status(self, losses: dict[str, torch.Tensor]):
        avg_return = np.mean(self.ep_returns[-10:]) if self.ep_returns else 0
        print(
            f"[{self.total_steps}] "
            f"AvgReturn: {avg_return:.1f} "
            + " ".join([f"{k}:{v.item():.4f}" for k, v in losses.items()])
        )

    def save_checkpoint(self):
        print(f"Saving checkpoint at step {self.total_steps}")

        # override in algorithm subclass if needed
        pass
