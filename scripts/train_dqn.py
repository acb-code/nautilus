import argparse, time, os
import gymnasium as gym
import numpy as np
from nautilus.core.buffers import ReplayBuffer
from nautilus.algos.dqn.agent import DQN

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', default='CartPole-v1')
    p.add_argument('--steps', type=int, default=50000)
    p.add_argument('--seed', type=int, default=1)
    args = p.parse_args()

    env = gym.make(args.env)
    rng = np.random.default_rng(args.seed)
    env.reset(seed=args.seed)
    obs, _ = env.reset()
    rb = ReplayBuffer(100000, env.observation_space.shape, (1,))
    agent = DQN(env.observation_space.shape[0], env.action_space.n)

    ep_ret, ep_len = 0.0, 0
    for t in range(1, args.steps + 1):
        a = agent.act(obs)
        nobs, r, term, trunc, _ = env.step(a)
        done = term or trunc
        rb.add(obs, [a], r, nobs, done)
        obs = nobs
        ep_ret += r; ep_len += 1

        if done:
            obs, _ = env.reset()
            ep_ret, ep_len = 0.0, 0

        if t > 1000:
            batch = rb.sample(64)
            loss = agent.update(batch)
        if t % 500 == 0:
            agent.sync()
            print(f"step={t}")

    print("Done.")

if __name__ == '__main__':
    main()
