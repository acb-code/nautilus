# Nautilus — Reinforcement Learning Examples

Learn-by-building RL algorithms with clean engineering: DQN and PPO first, then expand.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .[dev]
pre-commit install
```
Train a tiny DQN on CartPole:
```bash
python scripts/train_dqn.py --env CartPole-v1
```

See `nautilus/configs` for YAML examples.
