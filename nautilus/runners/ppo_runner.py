import argparse
import time

import gymnasium as gym
import numpy as np
import torch

from nautilus.algos.ppo.agent import PPOAgent
from nautilus.algos.ppo.config import PPOConfig
from nautilus.core.networks import ActorCritic, PixelActorCritic
from nautilus.utils.envs import make_atari_env, make_env


def parse_args():
    """
    Parses CLI arguments and overrides PPOConfig defaults.
    """
    parser = argparse.ArgumentParser(description="Nautilus PPO Runner")

    # Experiment Settings
    parser.add_argument(
        "--env-id", type=str, default="CartPole-v1", help="the id of the environment"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="run mode: train or test",
    )
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--num-envs", type=int, default=4, help="number of parallel environments")
    parser.add_argument(
        "--capture-video",
        action="store_true",
        help="whether to capture videos of the agent performances",
    )

    # Mapping args to Config fields (Simplified)
    parser.add_argument("--total-steps", type=int, default=500000)
    parser.add_argument("--lr", type=float, default=3e-4)

    # Logging Args
    parser.add_argument(
        "--track",
        action="store_true",
        help="if toggled, this experiment will be tracked with Weights and Biases",
    )
    parser.add_argument(
        "--wandb-project-name", type=str, default="nautilus-ppo", help="the wandb's project name"
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project"
    )

    # Load model args
    parser.add_argument(
        "--checkpoint", type=str, default="", help="path to a trained model.pt file"
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    run_name = f"{args.env_id}__{args.seed}__{int(time.time())}"

    # 1. Initialize Config
    config = PPOConfig(
        seed=args.seed,
        total_steps=args.total_steps,
        pi_lr=args.lr,
        vf_lr=args.lr,
        save_path=f"checkpoints/{run_name}",
        # Pass Logging Args
        track=args.track,
        wandb_project=args.wandb_project_name,
        wandb_entity=args.wandb_entity,
        run_name=run_name,
    )

    # 2. Setup Environment Factory
    # Detect if Atari/Pixels or Vector/State based on env ID or custom logic
    is_atari = "NoFrameskip" in args.env_id

    if is_atari:
        env_factory = make_atari_env
        NetworkClass = PixelActorCritic
    else:
        env_factory = make_env
        NetworkClass = ActorCritic  # Or ActorCriticShared if you prefer

    # 3. Create Vectorized Environments (The "MPI replacement")
    # SyncVectorEnv runs envs in serial (good for debugging/simple envs)
    # AsyncVectorEnv runs them in subprocesses (good for heavy envs)
    env_fns = [
        env_factory(args.env_id, args.seed + i, i, args.capture_video, run_name)
        for i in range(args.num_envs)
    ]
    # Use Async for parallelism if we have more than 1 env
    if args.num_envs > 1:
        envs = gym.vector.AsyncVectorEnv(env_fns)
    else:
        envs = gym.vector.SyncVectorEnv(env_fns)
    # 4. Setup Network
    # We create a dummy env just to get shapes for initialization
    dummy_env = envs.envs[0] if hasattr(envs, "envs") else gym.make(args.env_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = NetworkClass(dummy_env).to(device)

    # load checkpoint if specified
    if args.checkpoint:
        print(f"📂 Loading model from {args.checkpoint}...")
        network.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # 5. Setup Agent
    agent = PPOAgent(
        env_fn=lambda: envs,  # Agent expects a factory or instance
        actor_critic_module=network,
        config=config,
    )

    # 6. Execution
    if args.mode == "train":
        print(f"🚀 Starting training on {args.env_id} using {device}...")
        try:
            agent.train()
        except KeyboardInterrupt:
            print("Training interrupted. Saving checkpoint...")
            agent.save_checkpoint()

    elif args.mode == "test":
        print(f"🎥 Starting evaluation on {args.env_id}...")
        evaluate(agent, args.env_id, args.seed, run_name)

    envs.close()


def evaluate(agent, env_id, seed, run_name):
    """
    Runs a single instance of the environment for visualization/testing.
    """
    env = gym.make(env_id, render_mode="human")
    env = gym.wrappers.RecordEpisodeStatistics(env)

    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        # Select action (deterministic=True is often better for eval)
        # Note: Our current PPO select_action samples.
        # For rigorous eval, you might want to add a 'deterministic' flag to select_action.
        action, _ = agent.select_action(obs)

        # Handle vector-to-scalar action mismatch if necessary
        # (Our agent expects vector inputs/outputs usually)
        if isinstance(action, list | np.ndarray):
            step_action = action[0] if len(action.shape) > 0 else action
        else:
            step_action = action

        obs, _, terminated, truncated, info = env.step(step_action)
        done = terminated or truncated

        if done:
            print(f"Test Episode Return: {info['episode']['r']}")


if __name__ == "__main__":
    main()
