import os
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

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> tuple[Any, dict]:
        """
        Run the actor network to get action and auxiliary info (value, log_prob).
        """
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

            # Forward pass through actor
            dist = self.ac.pi(obs_tensor)
            if deterministic:
                # Discrete: take most likely action; Continuous: use mean
                action = dist.probs.argmax(dim=-1) if hasattr(dist, "probs") else dist.mean
                log_prob = dist.log_prob(action)
            else:
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
        # 1. Extract raw data (Time, Envs)
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        # Flatten observation dimensions for the network forward pass later
        # (T, N, ObsDim) -> (T*N, ObsDim)
        obs_flat = obs.view(-1, *obs.shape[2:])

        # Convert other numpy arrays
        rews = batch["rewards"]
        dones = batch["dones"].astype(np.float32)
        vals = np.array([i["val"] for i in batch["infos"]])  # Shape (T, N)

        # 2. Get Bootstrap Value
        with torch.no_grad():
            last_val = (
                self.ac.v(torch.as_tensor(self.obs, dtype=torch.float32, device=self.device))
                .cpu()
                .numpy()
            )

        # 3. Compute GAE (Vectorized) - Returns Numpy
        advs = compute_gae(
            rewards=rews,
            values=vals,
            dones=dones,
            gamma=self.config.gamma,
            lam=self.config.lam,
            next_value=last_val,
            next_done=np.zeros_like(last_val),
        )

        # 4. Compute Returns
        rets = advs + vals

        # 5. Flatten and Prepare Tensors

        # Actions
        actions = batch["actions"]
        if len(actions.shape) > 2:  # Continuous
            act_flat = torch.as_tensor(actions.reshape(-1, actions.shape[-1]), device=self.device)
        else:  # Discrete
            act_flat = torch.as_tensor(actions.flatten(), device=self.device)

        # LogProbs
        logprobs = np.array([i["log_prob"] for i in batch["infos"]])
        logp_old_flat = torch.as_tensor(logprobs.flatten(), dtype=torch.float32, device=self.device)

        # Returns
        ret_flat = torch.as_tensor(rets.flatten(), dtype=torch.float32, device=self.device)

        # --- FIX IS HERE ---
        # Normalize Advantages using Numpy BEFORE converting to Tensor
        adv_flat_np = advs.flatten()
        adv_flat_np = standardize_advantages(adv_flat_np)
        adv_flat = torch.as_tensor(adv_flat_np, dtype=torch.float32, device=self.device)

        return {
            "obs": obs_flat,
            "act": act_flat,
            "ret": ret_flat,
            "adv": adv_flat,
            "logp_old": logp_old_flat,
        }

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
        loss_pi = torch.tensor(0.0, device=self.device)
        kl = torch.tensor(0.0, device=self.device)
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

            # Add Entropy Bonus to encourage exploration
            ent = dist.entropy()
            if ent.dim() > 1:
                ent = ent.sum(-1)
            ent_bonus = 0.01 * ent.mean()
            loss_pi = loss_pi - ent_bonus

            loss_pi.backward()
            self.pi_optimizer.step()

        # --- Value Update ---
        loss_v = torch.tensor(0.0, device=self.device)
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

    def save_checkpoint(self):
        """
        Saves the PyTorch model to the path specified in config.
        """
        path = f"{self.config.save_path}.pt"

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save the Actor-Critic state dict
        torch.save(self.ac.state_dict(), path)
        print(f"💾 Model saved to {path}")
