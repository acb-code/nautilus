import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


def layer_init(layer, std=None, bias_const=0.0):
    """Standard Orthogonal initialization for RL stability"""
    if std is None:
        std = np.sqrt(2)
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MLP(nn.Module):
    """Simple Multi-Layer Perceptron backbone"""

    def __init__(self, input_dim, hidden_sizes, activation=nn.Tanh):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_sizes)
        for i in range(len(dims) - 1):
            layers.append(layer_init(nn.Linear(dims[i], dims[i + 1])))
            layers.append(activation())
        self.net = nn.Sequential(*layers)
        self.output_dim = dims[-1]

    def forward(self, x):
        return self.net(x)


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def forward(self, obs):
        return self._distribution(obs)


class GaussianActor(Actor):
    """
    Continuous Action Actor.
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.Tanh):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden_sizes, activation)

        # The "Mean" layer
        self.mu_layer = layer_init(nn.Linear(self.backbone.output_dim, act_dim), std=0.01)

        # The "Std" parameter (learnable, state-independent)
        # We initialize to -0.5 (approx 0.6 std) as in the original code
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)

    def _distribution(self, obs):
        features = self.backbone(obs)
        mu = self.mu_layer(features)

        # STABILITY FIX: Clamp the log_std to prevent numerical explosions
        # Original code used min_log_std=-20, max_log_std=2
        std = torch.exp(self.log_std.clamp(-20, 2))

        return Normal(mu, std)


class CategoricalActor(Actor):
    """Discrete Action Actor"""

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.Tanh):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden_sizes, activation)
        self.logits_layer = layer_init(nn.Linear(self.backbone.output_dim, act_dim), std=0.01)

    def _distribution(self, obs):
        features = self.backbone(obs)
        logits = self.logits_layer(features)
        return Categorical(logits=logits)


class Critic(nn.Module):
    """Value Network"""

    def __init__(self, obs_dim, hidden_sizes, activation=nn.Tanh):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden_sizes, activation)
        self.v_layer = layer_init(nn.Linear(self.backbone.output_dim, 1), std=1.0)

    def forward(self, obs):
        features = self.backbone(obs)
        return self.v_layer(features).squeeze(-1)


class ActorCritic(nn.Module):
    """
    Container class.
    """

    def __init__(self, env, hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        # Handle Gymnasium vs Gym dimension logic
        obs_dim = np.prod(env.observation_space.shape)

        # Determine Action Space
        try:
            # Continuous
            act_dim = env.action_space.shape[0]
            self.pi = GaussianActor(obs_dim, act_dim, hidden_sizes, activation)
        except (IndexError, AttributeError):
            # Discrete
            act_dim = env.action_space.n
            self.pi = CategoricalActor(obs_dim, act_dim, hidden_sizes, activation)

        self.v = Critic(obs_dim, hidden_sizes, activation)


class NatureCNN(nn.Module):
    """
    Standard DeepMind 'Nature' CNN architecture.
    Input: (Batch, Frames, 84, 84)
    Output: (Batch, 512) - The extracted feature vector.
    """

    def __init__(self, input_channels, features_dim=512):
        super().__init__()

        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Dynamic Calculation: Compute the size of the linear layer
        # by running a dummy forward pass.
        with torch.no_grad():
            # Create a dummy input: Batch size 1, and the correct channel/img dims
            dummy_input = torch.zeros(1, input_channels, 84, 84)
            n_flatten = self.cnn(dummy_input).shape[1]

        # The final layer maps the flattened pixels to a feature vector
        self.linear = nn.Sequential(
            layer_init(nn.Linear(n_flatten, features_dim)),
            nn.ReLU(),
        )

    def forward(self, x):
        # PRE-PROCESSING:
        # We assume input x is raw pixels (0-255).
        # We convert to float and normalize to [0, 1] here.
        # This prevents "forgetting to normalize" bugs in the runner.
        x = x.float() / 255.0

        x = self.cnn(x)
        x = self.linear(x)
        return x


class AtariActorCritic(nn.Module):
    """
    Actor-Critic architecture for Atari.
    - Shared CNN Backbone
    - Separate Actor and Critic Heads
    """

    def __init__(self, env, features_dim=512):
        super().__init__()

        # Auto-detect input channels (usually 4 for FrameStack)
        # Obs shape is typically (4, 84, 84)
        input_channels = env.observation_space.shape[0]
        n_actions = env.action_space.n

        # 1. Shared Backbone
        self.backbone = NatureCNN(input_channels, features_dim)

        # 2. Actor Head (Policy)
        # Maps features (512) -> Action Logits
        self.actor_head = layer_init(nn.Linear(features_dim, n_actions), std=0.01)

        # 3. Critic Head (Value)
        # Maps features (512) -> Value (1)
        self.critic_head = layer_init(nn.Linear(features_dim, 1), std=1.0)

    def get_value(self, x):
        """Helper to get just value (used in PPO update loop)"""
        features = self.backbone(x)
        return self.critic_head(features)

    def get_action_and_value(self, x, action=None):
        """
        Returns action, log_prob, entropy, and value.
        If action is provided, calculates log_prob/entropy for that specific action.
        If action is None, samples a new action.
        """
        features = self.backbone(x)

        # Actor Logic
        logits = self.actor_head(features)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        log_prob = probs.log_prob(action)
        entropy = probs.entropy()

        # Critic Logic
        value = self.critic_head(features)

        return action, log_prob, entropy, value


class PixelActorCritic(nn.Module):
    """
    Universal Agent for Pixel-based inputs.
    - Uses Shared CNN Backbone (NatureCNN)
    - Auto-detects Discrete vs Continuous action space
    """

    def __init__(self, env, features_dim=512):
        super().__init__()

        # 1. Setup Backbone
        obs_shape = env.observation_space.shape
        input_channels = obs_shape[0]  # Assumes (C, H, W)
        self.backbone = NatureCNN(input_channels, features_dim)

        # 2. Auto-detect Action Space
        if hasattr(env.action_space, "n"):
            self.is_continuous = False
            self.act_dim = env.action_space.n
        else:
            self.is_continuous = True
            self.act_dim = env.action_space.shape[0]

        # 3. Setup Heads
        if self.is_continuous:
            # --- Continuous (Gaussian) ---
            # Mean layer
            self.actor_mean = layer_init(nn.Linear(features_dim, self.act_dim), std=0.01)
            # Log Std (Learnable Parameter)
            self.actor_logstd = nn.Parameter(torch.zeros(1, self.act_dim) - 0.5)
        else:
            # --- Discrete (Categorical) ---
            self.actor_logits = layer_init(nn.Linear(features_dim, self.act_dim), std=0.01)

        # Critic is the same for both
        self.critic = layer_init(nn.Linear(features_dim, 1), std=1.0)

    def get_value(self, x):
        features = self.backbone(x)
        return self.critic(features)

    def get_action_and_value(self, x, action=None):
        features = self.backbone(x)

        # --- Distribution Logic ---
        if self.is_continuous:
            mu = self.actor_mean(features)
            # Clamp for stability (same as MLP version)
            std = torch.exp(self.actor_logstd.clamp(-20, 2))
            # Expand std to match batch size if needed (broadcasting handles this usually,
            # but being explicit is safe)
            probs = Normal(mu, std)
        else:
            logits = self.actor_logits(features)
            probs = Categorical(logits=logits)

        # --- Sampling ---
        if action is None:
            action = probs.sample()

        log_prob = probs.log_prob(action)
        entropy = probs.entropy()

        # For continuous, log_prob is summed across action dimensions
        if self.is_continuous:
            log_prob = log_prob.sum(1)
            entropy = entropy.sum(1)

        value = self.critic(features)

        return action, log_prob, entropy, value
