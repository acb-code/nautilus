# 🧭 Nautilus — Reinforcement Learning Examples

**Nautilus** is a reinforcement learning (RL) codebase.

---

## 🚀 Quickstart

```bash
# Create and activate a new environment
conda create -n nautilus python=3.11
conda activate nautilus

# Optional: install PyTorch with CUDA if available
# (Choose correct CUDA toolkit from https://pytorch.org/get-started/locally/)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Then install the repo and dev tools
pip install -e .[dev]
pre-commit install

## Optional for venv instead
# Create and activate environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install dependencies and dev tools
pip install -U pip
pip install -e .[dev]
pre-commit install
## End Optional
```

Run your first agent:

```bash
python scripts/train_dqn.py --env CartPole-v1
```

See progress with TensorBoard:

```bash
make tb
```

Logs, configs, and checkpoints are stored under:
```
runs/{algo}/{env}/{YYYYmmdd-HHMMSS}/
```

---

## 📂 Repository structure

```
nautilus/
  core/         # buffers, networks, samplers, advantages
  algos/        # implementations (dqn/, ppo/, tabular/)
  envs/         # gym + dm-control wrappers
  utils/        # logging, seeding, config, checkpointing
  runners/      # train loops and CLI entrypoints
  configs/      # YAML configs per algorithm/env
  tests/        # pytest suites
scripts/        # runnable scripts (train_dqn.py, train_ppo.py)
notebooks/      # learning notebooks & experiments
```

---

## 🧭 Learning roadmap

| Stage | Concepts | Implementation Targets |
|-------|-----------|------------------------|
| **M1 – Foundations** | MDPs, returns, buffers, exploration | utils/, buffers/, samplers/, basic train loop |
| **M2 – Bandits** | ε-greedy, UCB, regret | `algos/bandits/` |
| **M3 – Tabular Q-learning** | DP vs TD, off-policy updates | `algos/tabular/q_learning.py` |
| **M4 – Deep Q-Network (DQN)** | replay buffer, target net, ε-schedule | `algos/dqn/agent.py`, Atari wrappers |
| **M5 – Policy Gradients → PPO** | REINFORCE, GAE(λ), clipping, entropy bonus | `algos/ppo/agent.py` |
| **M6 – Extras** | Prioritized replay, n-step, distributed eval | `envs/`, `utils/`, `runners/` |

Each milestone comes with:
- Concept notebook (`notebooks/`)
- Unit tests (`tests/`)
- Reproducible configs (`configs/`)
- TensorBoard plots (`runs/`)

---

## 🧪 Development

Lint, format, and test:

```bash
make lint
make test
```

Run pre-commit hooks manually:

```bash
pre-commit run --all-files
```

---

## ⚙️ Configuration

All hyperparameters and environment settings live in `configs/`, e.g.:

```yaml
# configs/algos/dqn/cartpole.yaml
seed: 1
env: CartPole-v1
steps: 50000
batch_size: 64
gamma: 0.99
lr: 0.001
sync_interval: 500
```

CLI overrides work out of the box:

```bash
python scripts/train_dqn.py --env CartPole-v1 --steps 100000
```

---

## 📒 Learning resources

These implementations are inspired by:
- *Understanding Deep Learning* — Simon Prince (Chapter 19)
- Sutton & Barto — *Reinforcement Learning: An Introduction*
- OpenAI Spinning Up and CleanRL

The idea is to **re-implement, not copy**, so each concept is fully understood and engineered cleanly.

---

## 🧠 Road to mastery

Once DQN and PPO are solid, we’ll expand Nautilus to:
- Distributional & Dueling DQN, Noisy Nets
- SAC / TD3 for continuous control
- Multi-agent RL experiments
- LLM-driven agentic policy optimization
- Mixed-precision + MPI training

---

## 🤝 Contributing

Contributions, questions, and refactors are welcome.
Open an issue or PR — especially for docs, configs, or new environments.

If you’re learning RL: fork the repo, add your own experiments, and share results!

---

## 🪶 License

MIT License © 2025 Alexander Braafladt

---

### 🌊 “Build, test, understand — dive deeper.”
