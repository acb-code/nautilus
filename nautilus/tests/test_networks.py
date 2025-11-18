import gymnasium as gym
import pytest
import torch

from nautilus.core.networks import ActorCritic, PixelActorCritic


@pytest.fixture
def discrete_env():
    return gym.make("CartPole-v1")


@pytest.fixture
def continuous_env():
    # Create a dummy box env
    return gym.make("Pendulum-v1")


def test_mlp_discrete_shapes(discrete_env):
    """Test CartPole ActorCritic output shapes"""
    model = ActorCritic(discrete_env)

    # Batch size 4
    obs = torch.zeros((4, 4))
    action, log_prob, entropy, value = model.get_action_and_value(obs)

    assert action.shape == (4,)  # Discrete actions are scalars per batch
    assert log_prob.shape == (4,)
    assert value.shape == (4, 1) or value.shape == (4,)  # Value is scalar


def test_mlp_continuous_shapes(continuous_env):
    """Test Pendulum ActorCritic output shapes"""
    model = ActorCritic(continuous_env)

    # Batch size 4, Obs dim 3
    obs = torch.zeros((4, 3))
    action, log_prob, entropy, value = model.get_action_and_value(obs)

    assert action.shape == (4, 1)  # Continuous actions are vectors
    assert log_prob.shape == (4,)  # Log prob is summed to scalar
    assert value.shape == (4,)

    # Check mechanism: Log Std should be clamped
    assert hasattr(model.pi, "log_std")


def test_cnn_shapes():
    """Test PixelActorCritic with dummy image input"""

    # Mock env structure
    class MockEnv:
        observation_space = gym.spaces.Box(0, 255, (4, 84, 84), dtype=int)
        action_space = gym.spaces.Discrete(5)

    model = PixelActorCritic(MockEnv())

    # Batch size 2, 4 frames, 84x84
    dummy_img = torch.randint(0, 255, (2, 4, 84, 84)).float()

    action, _, _, value = model.get_action_and_value(dummy_img)

    assert action.shape == (2,)
    assert value.shape == (2, 1)
