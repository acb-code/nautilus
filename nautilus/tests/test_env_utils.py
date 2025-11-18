import gymnasium as gym
import numpy as np
from gymnasium.wrappers import ClipAction, RecordEpisodeStatistics

from nautilus.envs.api_compat import reset_env, step_env
from nautilus.utils.envs import make_env


class _OldGymStyleEnv:
    """Minimal env emulating Gym's pre-v0.26 API."""

    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def reset(self):
        return np.array([0.0], dtype=np.float32)

    def step(self, action):
        return np.array([1.0], dtype=np.float32), 1.0, True, {"terminal_called": True}


class _GymnasiumStyleEnv:
    """Minimal env emitting Gymnasium's reset/step signatures."""

    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def reset(self):
        return np.array([0.0], dtype=np.float32), {"reset_called": True}

    def step(self, action):
        return (
            np.array([1.0], dtype=np.float32),
            1.0,
            True,
            False,
            {"step_called": True},
        )


def _unwrap_types(env):
    """Collect wrapper class types from outermost to base env."""
    types = []
    current = env
    while True:
        types.append(type(current))
        if hasattr(current, "env"):
            current = current.env
        else:
            break
    return types


def test_reset_env_supports_old_and_new_gym_signatures():
    obs_old, info_old = reset_env(_OldGymStyleEnv())
    obs_new, info_new = reset_env(_GymnasiumStyleEnv())

    assert info_old == {}
    assert info_new == {"reset_called": True}
    np.testing.assert_array_equal(obs_old, np.array([0.0], dtype=np.float32))
    np.testing.assert_array_equal(obs_new, np.array([0.0], dtype=np.float32))


def test_step_env_handles_old_gym_and_manual_truncation():
    env = _OldGymStyleEnv()
    # Force a manual truncation to override returned "done"
    obs, reward, terminated, truncated, info = step_env(env, action=0, truncated=True)

    assert terminated is False
    assert truncated is True
    assert reward == 1.0
    np.testing.assert_array_equal(obs, np.array([1.0], dtype=np.float32))
    assert info["terminal_called"]


def test_step_env_handles_gymnasium_signature_and_truncation_passthrough():
    env = _GymnasiumStyleEnv()

    obs, reward, terminated, truncated, info = step_env(env, action=0, truncated=False)
    assert terminated is True
    assert truncated is False
    assert reward == 1.0
    assert info["step_called"]
    np.testing.assert_array_equal(obs, np.array([1.0], dtype=np.float32))

    # If caller flags truncation, it should suppress environment termination.
    obs2, _, terminated2, truncated2, _ = step_env(env, action=0, truncated=True)
    assert terminated2 is False
    assert truncated2 is True
    np.testing.assert_array_equal(obs2, np.array([1.0], dtype=np.float32))


def test_make_env_skips_clip_action_for_discrete_actions():
    thunk = make_env(
        env_id="CartPole-v1",
        seed=123,
        idx=0,
        capture_video=False,
        run_name="test-run",
        normalize=False,
    )
    env = thunk()
    try:
        wrapper_types = _unwrap_types(env)
        # RecordEpisodeStatistics should be applied even for discrete envs.
        assert any(t is RecordEpisodeStatistics for t in wrapper_types)
        # ClipAction should not be included for discrete action spaces.
        assert all(t is not ClipAction for t in wrapper_types)

        # Smoke test a single step to ensure wrapper stack works.
        obs, info = env.reset()
        action = env.action_space.sample()
        next_obs, _, terminated, truncated, _ = env.step(action)
        assert terminated is False or terminated is True  # sanity check bool-compatible
        assert truncated is False or truncated is True
        assert isinstance(next_obs, np.ndarray)
    finally:
        env.close()


def test_make_env_applies_clip_action_to_continuous_envs():
    thunk = make_env(
        env_id="Pendulum-v1",
        seed=123,
        idx=0,
        capture_video=False,
        run_name="test-run",
        normalize=False,
    )
    env = thunk()
    try:
        wrapper_types = _unwrap_types(env)
        assert any(t is ClipAction for t in wrapper_types)
    finally:
        env.close()
