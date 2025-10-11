
.PHONY: venv lint test tb dqn ppo

venv:
    python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e .[dev] && pre-commit install

lint:
    ruff check .
    ruff format --check .

test:
    pytest -q

tb:
    tensorboard --logdir runs

dqn:
    python scripts/train_dqn.py --env CartPole-v1

ppo:
    python scripts/train_ppo.py --env CartPole-v1
