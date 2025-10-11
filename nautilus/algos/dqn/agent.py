from __future__ import annotations
import torch, numpy as np, gymnasium as gym
from torch import nn, optim
from ...core.buffers import ReplayBuffer
from ...core.networks import MLP

class DQN:
    def __init__(self, obs_dim: int, act_dim: int, gamma=0.99, lr=1e-3, eps_start=1.0, eps_end=0.05, eps_decay=20000, device='cpu'):
        self.q = MLP(obs_dim, act_dim).to(device)
        self.target = MLP(obs_dim, act_dim).to(device)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.device = device
        self._steps = 0

    def act(self, obs):
        self._steps += 1
        eps = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self._steps / self.eps_decay)
        if np.random.rand() < eps:
            return np.random.randint(self.q.net[-1].out_features)
        with torch.no_grad():
            q = self.q(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            return int(q.argmax(dim=-1).item())

    def update(self, batch):
        obs, act, rew, nxt, done = batch
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(act, dtype=torch.long, device=self.device).view(-1)
        rew = torch.as_tensor(rew, dtype=torch.float32, device=self.device)
        nxt = torch.as_tensor(nxt, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device)

        q = self.q(obs).gather(1, act.view(-1,1)).squeeze(1)
        with torch.no_grad():
            max_next = self.target(nxt).max(dim=1).values
            target = rew + self.gamma * (1.0 - done) * max_next
        loss = nn.functional.mse_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())

    def sync(self, tau=1.0):
        if tau >= 1.0:
            self.target.load_state_dict(self.q.state_dict())
        else:
            for tp, p in zip(self.target.parameters(), self.q.parameters()):
                tp.data.copy_(tau * tp.data + (1 - tau) * p.data)
