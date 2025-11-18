import gymnasium as gym
import numpy as np
from gymnasium.wrappers import (
    ClipAction,
    NormalizeObservation,
    NormalizeReward,
    RecordVideo,
    TransformObservation,
)


def make_env(env_id, seed, idx, capture_video, run_name, gamma=0.99):
    """
    Factory function to create a single environment instance.
    """

    def thunk():
        # 1. Create Base Env
        render_mode = "rgb_array" if capture_video and idx == 0 else None
        env = gym.make(env_id, render_mode=render_mode)

        if capture_video and idx == 0:
            env = RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda x: x % 1000 == 0)

        # 2. Standard Wrappers
        env = gym.wrappers.RecordEpisodeStatistics(env)

        # FIX 1: Only use ClipAction for Continuous (Box) environments
        # CartPole is Discrete, so this wrapper would crash it.
        if isinstance(env.action_space, gym.spaces.Box):
            env = ClipAction(env)

        # 3. Normalization
        env = NormalizeObservation(env)
        env = NormalizeReward(env, gamma=gamma)

        # FIX 2: Explicitly pass observation_space to TransformObservation
        # Newer Gymnasium versions require this argument.
        env = TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)

        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def make_atari_env(env_id, seed, idx, capture_video, run_name):
    """
    Specific factory for Atari (Pixels).
    """

    def thunk():
        render_mode = "rgb_array" if capture_video and idx == 0 else None
        env = gym.make(env_id, render_mode=render_mode)

        if capture_video and idx == 0:
            env = RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda x: x % 1000 == 0)

        # Standard Atari Wrappers
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.AtariPreprocessing(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk
