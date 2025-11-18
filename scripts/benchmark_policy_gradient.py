import contextlib

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from nautilus.algos.ppo.agent import PPOAgent
from nautilus.algos.ppo.config import PPOConfig
from nautilus.algos.vpg.agent import VPGAgent
from nautilus.core.networks import ActorCritic
from nautilus.utils.envs import make_env


def run_experiment(algo_name, agent_cls, total_steps=200000, seed=42):
    print(f"🚀 Benchmarking {algo_name}...")

    # 1. Setup
    env_id = "CartPole-v1"
    num_envs = 4
    run_name = f"{algo_name}_{seed}"

    # Config
    config = PPOConfig(
        seed=seed,
        total_steps=total_steps,
        pi_lr=3e-4,
        vf_lr=1e-3,
        # PPO specific (VPG will ignore these)
        train_pi_iter=4,
        clip_ratio=0.2,
    )

    # Envs
    env_fns = [make_env(env_id, seed + i, i, False, run_name) for i in range(num_envs)]
    envs = gym.vector.SyncVectorEnv(env_fns)

    # Network & Agent
    device = torch.device("cpu")
    dummy_env = gym.make(env_id)
    network = ActorCritic(dummy_env).to(device)

    agent = agent_cls(lambda: envs, network, config)

    # 2. Train Loop with Data Collection
    with contextlib.suppress(KeyboardInterrupt):
        # Manually run loop to capture returns
        agent.train()
        # Extract logs from agent history (we need to hack the logger slightly
        # or just use the returns stored in the agent)

        # For this script, we'll look at the ep_returns buffer in the agent
        # In a real scenario, we'd parse the TensorBoard event file.
        # Let's assume PolicyOptimizerBase tracks ep_returns.

    envs.close()

    # 3. Process Data
    # We reconstruct a timeline from the recorded lengths
    cum_steps = 0
    data = []
    for r, ep_len in zip(agent.ep_returns, agent.ep_lengths, strict=False):
        # FIX: Handle cases where vector envs bundle multiple episode stats
        # We treat 'r' and 'l' as arrays (even if they are scalars) and iterate
        r_list = np.atleast_1d(r)
        l_list = np.atleast_1d(ep_len)

        for i in range(len(r_list)):
            # Safely extract item
            safe_r = (
                float(np.mean(r_list[i])) if hasattr(r_list[i], "__len__") else float(r_list[i])
            )
            safe_l = (
                float(np.mean(l_list[i])) if hasattr(l_list[i], "__len__") else float(l_list[i])
            )

            cum_steps += safe_l
            data.append({"step": cum_steps, "return": safe_r, "algorithm": algo_name, "seed": seed})

    return data


def plot_results(all_data):
    df = pd.DataFrame(all_data)

    # --- THE FIX: Apply Rolling Average ---
    # We group by seed and algorithm, then smooth the returns over 50 episodes
    df["smoothed_return"] = df.groupby(["algorithm", "seed"])["return"].transform(
        lambda x: x.rolling(window=100, min_periods=1).mean()
    )

    plt.figure(figsize=(10, 6))

    # Plot the Smoothed Return instead of raw 'return'
    sns.lineplot(data=df, x="step", y="smoothed_return", hue="algorithm")

    plt.title("PPO vs VPG (Smoothed window=50)")
    plt.xlabel("Environment Steps")
    plt.ylabel("Episode Return (Smoothed)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("benchmark_results_smoothed.png")
    print("✅ Plot saved to benchmark_results_smoothed.png")
    plt.show()


if __name__ == "__main__":
    experiments = [("PPO", PPOAgent), ("VPG", VPGAgent)]

    all_results = []

    # Run for 2 seeds to get a standard deviation ribbon
    for seed in [1, 2]:
        for name, cls in experiments:
            data = run_experiment(name, cls, total_steps=200000, seed=seed)
            all_results.extend(data)

    plot_results(all_results)
