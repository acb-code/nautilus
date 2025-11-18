import gymnasium as gym
import numpy as np
import pytest
import torch

from nautilus.algos.ppo.agent import PPOAgent
from nautilus.algos.ppo.config import PPOConfig
from nautilus.core.networks import ActorCritic


def test_agent_update_step():
    """
    Integration Test: Run one optimization step and ensure weights change.
    """
    # 1. Setup
    env_id = "CartPole-v1"

    # Mock factory function
    def env_fn():
        env = gym.make(env_id)
        env.num_envs = 1  # Pretend it's vectorized
        return env

    # Use a large target_kl to avoid early stopping on synthetic data.
    config = PPOConfig(total_steps=100, train_pi_iter=1, train_v_iter=1, target_kl=1e6)
    dummy_env = gym.make(env_id)
    network = ActorCritic(dummy_env)

    agent = PPOAgent(env_fn, network, config)

    # 2. Create Fake Batch Data
    # We need T=10, N=1
    batch_size = 10
    obs_dim = dummy_env.observation_space.shape[0]

    batch = {
        "obs": np.random.randn(batch_size, 1, obs_dim).astype(np.float32),
        "actions": np.random.randint(0, 2, (batch_size, 1)),
        "rewards": np.random.randn(batch_size, 1),
        "dones": np.zeros((batch_size, 1)),
        # PPOAgent expects value/log_prob arrays shaped like the vectorized env output.
        "infos": [
            {"val": np.array([0.0]), "log_prob": np.array([-0.6])} for _ in range(batch_size)
        ],
    }

    # 3. Snapshot weights before update
    old_weight = agent.ac.pi.backbone.net[0].weight.data.clone()

    # 4. Run Update
    losses = agent.compute_losses(batch)
    agent.update_params(losses)

    # 5. Check weights changed
    new_weight = agent.ac.pi.backbone.net[0].weight.data
    assert not torch.allclose(old_weight, new_weight), "Weights did not update!"


def test_agent_update_early_stop_logs_metrics():
    """
    Ensure metrics are populated even when early stopping prevents an update step.
    """
    env_id = "CartPole-v1"

    def env_fn():
        env = gym.make(env_id)
        env.num_envs = 1
        return env

    # Force early stopping via tiny KL threshold
    config = PPOConfig(total_steps=10, train_pi_iter=1, train_v_iter=0, target_kl=1e-8)
    dummy_env = gym.make(env_id)
    network = ActorCritic(dummy_env)
    agent = PPOAgent(env_fn, network, config)

    batch_size = 2
    obs_dim = dummy_env.observation_space.shape[0]
    batch = {
        "obs": np.random.randn(batch_size, 1, obs_dim).astype(np.float32),
        "actions": np.random.randint(0, 2, (batch_size, 1)),
        "rewards": np.random.randn(batch_size, 1),
        "dones": np.zeros((batch_size, 1)),
        "infos": [
            {"val": np.array([0.0]), "log_prob": np.array([-0.6])} for _ in range(batch_size)
        ],
    }

    old_weight = agent.ac.pi.backbone.net[0].weight.data.clone()

    losses = agent.compute_losses(batch)
    agent.update_params(losses)

    # Early stop should leave weights untouched, but metrics should still exist
    new_weight = agent.ac.pi.backbone.net[0].weight.data
    assert torch.allclose(old_weight, new_weight)
    assert hasattr(agent, "latest_metrics")
    for key in ["loss_pi", "loss_v", "kl"]:
        assert key in agent.latest_metrics


def test_agent_lr_decay_updates_optimizers():
    """
    Learning rates should linearly decay based on training progress when enabled.
    """
    env_id = "CartPole-v1"

    def env_fn():
        env = gym.make(env_id)
        env.num_envs = 1
        return env

    total_steps = 100
    config = PPOConfig(
        total_steps=total_steps,
        train_pi_iter=1,
        train_v_iter=0,
        target_kl=1e6,
        lr_decay=True,
    )
    dummy_env = gym.make(env_id)
    network = ActorCritic(dummy_env)
    agent = PPOAgent(env_fn, network, config)

    batch_size = 2
    obs_dim = dummy_env.observation_space.shape[0]
    batch = {
        "obs": np.random.randn(batch_size, 1, obs_dim).astype(np.float32),
        "actions": np.random.randint(0, 2, (batch_size, 1)),
        "rewards": np.random.randn(batch_size, 1),
        "dones": np.zeros((batch_size, 1)),
        "infos": [
            {"val": np.array([0.0]), "log_prob": np.array([-0.6])} for _ in range(batch_size)
        ],
    }

    # Simulate mid-training progress before the update step.
    agent.total_steps = total_steps // 2

    initial_pi_lr = agent.pi_optimizer.param_groups[0]["lr"]
    losses = agent.compute_losses(batch)
    agent.update_params(losses)
    updated_pi_lr = agent.pi_optimizer.param_groups[0]["lr"]

    expected_lr = initial_pi_lr * 0.5
    assert pytest.approx(updated_pi_lr, rel=1e-6, abs=0.0) == expected_lr
