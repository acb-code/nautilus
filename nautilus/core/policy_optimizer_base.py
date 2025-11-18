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

from nautilus.core.on_policy import OnPolicyBuffer
from nautilus.envs.api_compat import reset_env, step_env
from nautilus.utils.logger import Logger


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

    # --- Logging Params ---
    track: bool = False
    wandb_project: str = "nautilus-project"
    wandb_entity: str = None
    run_name: str = "test_run"


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
        self.rollout_buffer = OnPolicyBuffer(config.rollout_length, self.num_envs)

        self._seed_everything(config.seed)

        self.total_steps = 0

        # Logs
        self.ep_returns = []
        self.ep_lengths = []

        # Setup Logger
        self.logger = Logger(
            log_dir=f"runs/{config.run_name}",
            config=config,
            use_wandb=config.track,
            run_name=config.run_name,
        )

        # Track Steps Per Second (SPS)
        self.start_time = 0

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
        self.rollout_buffer.reset()

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
            self.rollout_buffer.add(
                obs=self.obs,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )

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

        return self.rollout_buffer.get()

    def train(self):
        import time

        self.start_time = time.time()  # Start timer

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
        import time

        # 1. Calculate Performance Metrics
        avg_return = np.mean(self.ep_returns[-100:]) if self.ep_returns else 0
        avg_length = np.mean(self.ep_lengths[-100:]) if self.ep_lengths else 0
        sps = int(self.total_steps / (time.time() - self.start_time))

        # 2. Prepare Dictionary
        metrics = {
            "charts/episodic_return": avg_return,
            "charts/episodic_length": avg_length,
            "charts/SPS": sps,
        }

        # Add Algorithm specific losses (flattened)
        for k, v in losses.items():
            if isinstance(v, float | int):
                metrics[f"losses/{k}"] = v
            elif hasattr(v, "item"):
                metrics[f"losses/{k}"] = v.item()

        # 3. Log to Backend
        self.logger.log(metrics, self.total_steps)

        # 4. Print to Console (for sanity check)
        print(
            f"[{self.total_steps}] Return: {avg_return:.2f} | SPS: {sps} | "
            + " ".join([f"{k}:{v:.3f}" for k, v in losses.items() if isinstance(v, float | int)])
        )

    def save_checkpoint(self):
        pass
