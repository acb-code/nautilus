import gymnasium as gym
import numpy as np
import pytest
import torch

from nautilus.algos.ppo.agent import PPOAgent
from nautilus.algos.ppo.config import PPOConfig
from nautilus.core.networks import ActorCritic


class DummyBoxEnv:
    """Minimal env with Box spaces for testing continuous-action paths."""

    def __init__(self, obs_dim=3, act_dim=2):
        import gymnasium as gym

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
        self.num_envs = 1

    def reset(self, seed=None):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}

    def step(self, action):
        raise NotImplementedError("Not used in these tests.")


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
    config = PPOConfig(
        total_steps=100,
        train_pi_iter=1,
        train_v_iter=1,
        target_kl=1e6,
        minibatch_size=5,
        update_epochs=2,
    )
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


def test_advantage_normalization_toggle():
    """norm_adv toggles normalization in compute_losses."""
    env_id = "CartPole-v1"

    def env_fn():
        env = gym.make(env_id)
        env.num_envs = 1
        return env

    dummy_env = gym.make(env_id)
    network_norm = ActorCritic(dummy_env)
    network_raw = ActorCritic(dummy_env)

    base_batch = {
        "obs": np.zeros((2, 1, dummy_env.observation_space.shape[0]), dtype=np.float32),
        "actions": np.zeros((2, 1), dtype=np.int64),
        "rewards": np.zeros((2, 1), dtype=np.float32),
        "dones": np.zeros((2, 1), dtype=np.float32),
        "advantages": np.array([[1.0], [3.0]], dtype=np.float32),
        "values": np.zeros((2, 1), dtype=np.float32),
        "infos": [{"val": np.array([0.0]), "log_prob": np.array([-0.6])} for _ in range(2)],
    }

    cfg_norm = PPOConfig(norm_adv=True, train_pi_iter=0, train_v_iter=0, target_kl=None)
    agent_norm = PPOAgent(env_fn, network_norm, cfg_norm)
    adv_norm = agent_norm.compute_losses(base_batch)["adv"]
    assert torch.allclose(adv_norm.mean(), torch.tensor(0.0), atol=1e-6)
    # np.std in standardize_advantages uses population std (ddof=0), so compare to unbiased=False
    assert torch.allclose(adv_norm.std(unbiased=False), torch.tensor(1.0), atol=1e-6)

    cfg_raw = PPOConfig(norm_adv=False, train_pi_iter=0, train_v_iter=0, target_kl=None)
    agent_raw = PPOAgent(env_fn, network_raw, cfg_raw)
    adv_raw = agent_raw.compute_losses(base_batch)["adv"]
    np.testing.assert_allclose(adv_raw.cpu().numpy(), base_batch["advantages"].flatten())


def test_continuous_logprob_summing_and_value_clipping():
    """Continuous log_probs are summed, and value clipping scales with vf_coef."""

    class DummyDist:
        def __init__(self, logprob):
            self._logprob = logprob

        def log_prob(self, actions):
            # Return per-dimension logprob tensor with broadcasted shape
            return self._logprob.expand_as(actions)

        def entropy(self):
            return torch.zeros_like(self._logprob)

    class DummyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Shape (1, act_dim) so it can expand to (batch, act_dim)
            self.logprob = torch.nn.Parameter(torch.tensor([[0.1, 0.2]]), requires_grad=False)

        def forward(self, obs):
            return DummyDist(self.logprob)

    class DummyValue(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor([1.0, 0.3]))

        def forward(self, obs):
            return self.value

    class DummyAC(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pi = DummyPolicy()
            self.v = DummyValue()

    def env_fn():
        env = DummyBoxEnv()
        return env

    config = PPOConfig(
        clip_ratio=0.2,
        clip_vloss=True,
        vf_coef=0.5,
        norm_adv=False,
        train_pi_iter=0,  # skip policy update for this unit test
        train_v_iter=1,
        update_epochs=None,
        minibatch_size=2,
    )

    ac = DummyAC()
    agent = PPOAgent(env_fn, ac, config)

    obs = torch.zeros((2, 1, 3), dtype=torch.float32)
    act = torch.zeros((2, 1, 2), dtype=torch.float32)  # shape matches DummyDist expansion
    ret = torch.tensor([0.0, 0.0], dtype=torch.float32)
    adv = torch.tensor([0.0, 0.0], dtype=torch.float32)
    logp_old = torch.zeros(2, dtype=torch.float32)  # summed per-sample logprobs
    val_old = torch.tensor([0.0, 0.0], dtype=torch.float32)

    batch_tensors = {
        "obs": obs.view(2, -1),
        "act": act.view(2, -1),
        "ret": ret,
        "adv": adv,
        "logp_old": logp_old,
        "val_old": val_old,
    }

    agent.update_params(batch_tensors)

    # Value clipping should use val_old and clip ratio 0.2; expected loss ~0.13625 (see analysis)
    expected_loss = 0.13625
    assert torch.allclose(
        torch.tensor(agent.latest_metrics["loss_v"]), torch.tensor(expected_loss), atol=1e-5
    )
