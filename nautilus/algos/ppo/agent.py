from typing import Any

import numpy as np
import torch
import torch.optim as optim

from nautilus.algos.ppo.config import PPOConfig
from nautilus.algos.ppo.losses import (
    approximate_kl,
    ppo_policy_loss,
    value_loss,
)

# Import the functional utilities you defined previously
from nautilus.core.on_policy import (
    compute_gae,
    standardize_advantages,
)
from nautilus.core.policy_optimizer_base import PolicyOptimizerBase


class PPOAgent(PolicyOptimizerBase):
    def __init__(self, env_fn, actor_critic_module, config: PPOConfig):
        """
        Args:
            env_fn: Function creating the gym environment.
            actor_critic_module: A torch.nn.Module containing .pi (actor) and .v (critic).
            config: PPOConfig object.
        """
        super().__init__(env_fn, config)
        self.config: PPOConfig = config

        # Move model to device
        self.ac = actor_critic_module.to(self.device)

        # Set up optimizers
        self.pi_optimizer = optim.Adam(self.ac.pi.parameters(), lr=config.pi_lr)
        self.v_optimizer = optim.Adam(self.ac.v.parameters(), lr=config.vf_lr)

    def select_action(self, obs: np.ndarray) -> tuple[Any, dict]:
        """
        Run the actor network to get action and auxiliary info (value, log_prob).
        """
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

            # Forward pass through actor
            dist = self.ac.pi(obs_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Forward pass through critic (needed for GAE later)
            value = self.ac.v(obs_tensor)

        # Return numpy arrays
        return action.cpu().numpy(), {
            "val": value.cpu().numpy(),
            "log_prob": log_prob.cpu().numpy(),
        }

    def compute_losses(self, batch: dict) -> dict:
        """
        Note: In PPO, we don't compute a single loss and return it.
        We iterate multiple times. This method is used here to PRE-PROCESS
        the batch (calculate GAE) before the update loop.
        """
        # 1. Extract raw data
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        rews = batch["rewards"]  # numpy
        dones = batch["dones"]  # numpy
        vals = np.array([i["val"] for i in batch["infos"]])  # Extract values from info dict

        # 2. Calculate GAE and Returns (using functional utils)
        # We need the value of the "next" state after the rollout for bootstrapping
        with torch.no_grad():
            last_val = (
                self.ac.v(torch.as_tensor(self.obs, dtype=torch.float32, device=self.device))
                .cpu()
                .numpy()
            )

        # Append last_val to values for GAE calculation
        bootstrap_values = [last_val]  # Simplified: assumes 1 env/episode structure for now

        # Compute Advantages (GAE)
        advs = compute_gae(
            rewards=rews,
            values=vals,
            dones=dones,
            gamma=self.config.gamma,
            lam=self.config.lam,
            bootstrap_values=bootstrap_values,  # This assumes strict episodic structure, see note below
        )

        # Compute Returns (Ret = Adv + Val)
        rets = advs + vals

        # Standardize Advantages
        advs = standardize_advantages(advs)

        # Convert to Torch
        batch_tensors = {
            "obs": obs,
            "act": torch.as_tensor(batch["actions"], device=self.device),
            "ret": torch.as_tensor(rets, dtype=torch.float32, device=self.device),
            "adv": torch.as_tensor(advs, dtype=torch.float32, device=self.device),
            "logp_old": torch.as_tensor(
                [i["log_prob"] for i in batch["infos"]], dtype=torch.float32, device=self.device
            ),
        }

        return batch_tensors

    def update_params(self, batch_tensors: dict):
        """
        PPO Update Loop:
        1. Run policy gradient descent for `train_pi_iter` steps.
        2. Run value function descent for `train_v_iter` steps.
        """
        obs = batch_tensors["obs"]
        act = batch_tensors["act"]
        ret = batch_tensors["ret"]
        adv = batch_tensors["adv"]
        logp_old = batch_tensors["logp_old"]

        # --- Policy Update ---
        for i in range(self.config.train_pi_iter):
            self.pi_optimizer.zero_grad()

            # Get current distribution
            dist = self.ac.pi(obs)
            logp_new = dist.log_prob(act)

            # Check KL for early stopping
            kl = approximate_kl(logp_old, logp_new)
            if kl > 1.5 * self.config.target_kl:
                print(f"Early stopping at step {i} due to reaching max KL.")
                break

            # Calculate PPO Loss (Functional)
            loss_pi = ppo_policy_loss(
                new_log_probs=logp_new,
                old_log_probs=logp_old,
                advantages=adv,
                clip_ratio=self.config.clip_ratio,
            )

            # Add Entropy Bonus (Optional)
            # ent = dist.entropy().mean()
            # loss_pi = loss_pi - 0.01 * ent

            loss_pi.backward()
            self.pi_optimizer.step()

        # --- Value Update ---
        for _i in range(self.config.train_v_iter):
            self.v_optimizer.zero_grad()

            pred_val = self.ac.v(obs)

            # Calculate Value Loss (Functional)
            loss_v = value_loss(
                values=pred_val,
                target_values=ret,
                clip=None,  # Optional: implement value clipping here
            )

            loss_v.backward()
            self.v_optimizer.step()

        # Store metrics for logging (optional)
        self.latest_metrics = {
            "loss_pi": loss_pi.item(),
            "loss_v": loss_v.item(),
            "kl": kl.item(),
            "delta_loss_pi": (loss_pi.item() - 0),  # simplified
        }

    def log_status(self, losses: dict):
        # Override to log the metrics generated during update_params
        if hasattr(self, "latest_metrics"):
            super().log_status(self.latest_metrics)
