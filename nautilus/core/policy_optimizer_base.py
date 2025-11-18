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
    def __init__(self, env_fn, config: OptimizerConfig):
        self.env_fn = env_fn
        self.config = config
        self._setup_backend()

        self.device = torch.device(config.device)

        # Environment setup
        self.env = env_fn()
        self.obs, _ = reset_env(self.env)

        # Determine number of environments (1 if standard, N if vectorized)
        self.num_envs = getattr(self.env, "num_envs", 1)

        self._seed_everything(config.seed)

        self.total_steps = 0

        # Logs
        self.ep_returns = []
        self.ep_lengths = []

    def _setup_backend(self):
        if self.config.backend == "torch":
            torch.set_num_threads(1)

    def _seed_everything(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        # VectorEnvs handle seeding via action_space usually,
        # but robust seeding is handled in the env factory.
        pass

    def select_action(self, obs: np.ndarray) -> tuple[Any, dict]:
        raise NotImplementedError

    def compute_losses(self, batch: dict) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def update_params(self, losses: dict[str, torch.Tensor]):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Vector-Safe Rollout Collection
    # ------------------------------------------------------------------

    def collect_rollout(self) -> dict[str, np.ndarray]:
        """
        Collects a rollout. Handles both Vectorized and Standard environments.
        """
        obs_buf = []
        actions_buf = []
        rewards_buf = []
        dones_buf = []
        infos_buf = []

        for _t in range(self.config.rollout_length):
            # 1. Select Action
            action, info = self.select_action(self.obs)

            # 2. Step Environment
            next_obs, reward, terminated, truncated, env_info = step_env(
                self.env, action, truncated=False
            )

            # FIX 1: Vector-safe Boolean Logic
            # 'terminated' and 'truncated' are arrays if num_envs > 1
            done = np.logical_or(terminated, truncated)

            # 3. Store Data
            obs_buf.append(self.obs)
            actions_buf.append(action)
            rewards_buf.append(reward)
            dones_buf.append(done)
            infos_buf.append(info)

            # FIX 2: Log Episode Stats via 'final_info'
            # Gymnasium VectorEnvs auto-reset. When an episode finishes,
            # the stats are hidden inside 'final_info' or 'episode' dicts.
            if "final_info" in env_info:
                for final_item in env_info["final_info"]:
                    # 'final_item' is None if that specific env didn't finish
                    if final_item is not None and "episode" in final_item:
                        self.ep_returns.append(final_item["episode"]["r"])
                        self.ep_lengths.append(final_item["episode"]["l"])

            # Fallback for non-vectorized envs or older wrappers
            elif "episode" in env_info:
                self.ep_returns.append(env_info["episode"]["r"])
                self.ep_lengths.append(env_info["episode"]["l"])

            # FIX 3: Handle Auto-Reset
            # VectorEnvs automatically reset done environments.
            # 'next_obs' is already the reset observation for those envs.
            self.obs = next_obs

            # Increment total steps by the number of parallel environments
            self.total_steps += self.num_envs

        return dict(
            obs=np.array(obs_buf),
            actions=np.array(actions_buf),
            rewards=np.array(rewards_buf),
            dones=np.array(dones_buf),
            infos=infos_buf,
        )

    def train(self):
        while self.total_steps < self.config.total_steps:
            batch = self.collect_rollout()
            losses = self.compute_losses(batch)
            self.update_params(losses)

            if (
                self.total_steps % self.config.log_interval
                < self.config.rollout_length * self.num_envs
            ):
                self.log_status(losses)

            if (
                self.total_steps % self.config.save_interval
                < self.config.rollout_length * self.num_envs
            ):
                self.save_checkpoint()

    def log_status(self, losses: dict[str, torch.Tensor]):
        avg_return = np.mean(self.ep_returns[-10:]) if self.ep_returns else 0
        loss_str = " ".join([
            f"{k}:{v:.4f}" for k, v in losses.items() if isinstance(v, float | int)
        ])
        print(f"[{self.total_steps}] AvgReturn: {avg_return:.1f} {loss_str}")

    def save_checkpoint(self):
        pass
