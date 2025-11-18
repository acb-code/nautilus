"""
Pure functional utilities for on-policy RL algorithms.

These functions implement:
- discounted returns (full-episode or reward-to-go)
- generalized advantage estimation (GAE)
- advantage normalization

All functions are numpy-only and free of side effects.
"""

import jax.lax as lax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# 1. Returns / Discounted Reward Computations
# ---------------------------------------------------------------------------


def compute_returns_full_episode(
    rewards: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Compute *full-episode* discounted return R for every timestep:
        R_t = sum_{k=t..T} gamma^(k-t) * r_k
    But full-episode means we actually compute the *same* total return
    and assign it to each timestep.
    """
    T = len(rewards)
    indices = np.arange(T)
    discounts = np.power(gamma, indices)
    total_return = np.sum(rewards * discounts)
    return np.ones(T) * total_return


def compute_returns_reward_to_go(
    rewards: np.ndarray,
    gamma: float,
    bootstrap: float = 0.0,
) -> np.ndarray:
    """
    Compute reward-to-go returns:
        R_t = r_t + gamma r_{t+1} + ... + gamma^(n-t) * bootstrap

    Args:
        rewards: numpy array of episode rewards.
        gamma: discount factor.
        bootstrap: final value estimate V(s_T) or 0 if no value function.

    Returns:
        returns: array of reward-to-go values.
    """
    T = len(rewards)
    returns = np.zeros(T + 1)
    returns[-1] = bootstrap

    # Reverse-time accumulate
    for t in reversed(range(T)):
        returns[t] = rewards[t] + gamma * returns[t + 1]

    return returns[:-1]


# ---------------------------------------------------------------------------
# 2. GAE (Generalized Advantage Estimation)
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
    bootstrap_values: np.ndarray,
) -> np.ndarray:
    """
    Compute GAE over a *sequence of episodes* contained in a single rollout.

    Args:
        rewards: shape [T]
        values: shape [T]
        dones: boolean array shape [T] marking terminal timesteps
        gamma: discount factor
        lam: lambda for GAE
        bootstrap_values: array of value estimates at episode boundaries.
                          Typically one per episode.

    Notes:
        - We loop over episodes separated by dones.
        - For each episode, compute:
            delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            advantage_t = sum_{k=t..T} (gamma*lam)^(k-t) * delta_k

    Returns:
        advantages: numpy array shape [T]
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)

    # find episodes: indices where done=True
    terminals = np.where(dones.astype(int) == 1)[0]
    terminals = np.concatenate(([-1], terminals))  # include start boundary

    ep_idx = 0
    for ep_idx, (t0, t1) in enumerate(zip(terminals[:-1], terminals[1:], strict=False)):
        start = t0 + 1
        end = t1 + 1
        bootstrap_v = bootstrap_values[ep_idx]

        ep_rewards = rewards[start:end]
        ep_values = values[start:end]
        ep_next_values = np.concatenate((ep_values[1:], [bootstrap_v]))

        # delta_t = r_t + gamma V_{t+1} - V_t
        deltas = ep_rewards + gamma * ep_next_values - ep_values

        # compute discounted sum of deltas backwards
        gae = 0.0
        for t in reversed(range(len(deltas))):
            gae = deltas[t] + gamma * lam * gae
            advantages[start + t] = gae

    return advantages


# ---------------------------------------------------------------------------
# 3. Advantage normalization
# ---------------------------------------------------------------------------


def standardize_advantages(advantages: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Normalize advantages to have zero mean and unit variance.
    """
    mean = advantages.mean()
    std = advantages.std()
    return (advantages - mean) / (std + eps)


# ---------------------------------------------------------------------------
# 1. Returns / Discounted Reward Computations (JAX)
# ---------------------------------------------------------------------------


def compute_returns_full_episode_jax(
    rewards: jnp.ndarray,
    gamma: float,
) -> jnp.ndarray:
    """
    Compute *full-episode* discounted return R for every timestep. (JAX implementation)
    """
    T = len(rewards)
    indices = jnp.arange(T)
    discounts = jnp.power(gamma, indices)
    total_return = jnp.sum(rewards * discounts)
    return jnp.ones(T) * total_return


def compute_returns_reward_to_go_jax(
    rewards: jnp.ndarray,
    gamma: float,
    bootstrap: float = 0.0,
) -> jnp.ndarray:
    """
    Compute reward-to-go returns using an efficient array operation. (JAX implementation)
    This avoids the Python `for` loop in the original NumPy version
    by using `jax.lax.scan` or a similar convolutional approach,
    but for simplicity and direct translation, we can use a reverse cumulative sum.

    A more JAX-idiomatic approach for the loop is often preferred for JIT compatibility.
    We will use a functional approach here.
    """
    T = len(rewards)

    # 1. Reverse rewards and apply discount
    # The required operation is a discounted *suffix* sum, which is a scan operation.

    # Simple version (using a loop which is fine if not jitted, but less JAX-idiomatic)
    # A true JAX version would use jax.lax.scan or a specialized convolution.

    returns = jnp.zeros(T + 1)
    returns = returns.at[-1].set(bootstrap)

    # JAX friendly alternative (requires manual scan/associative scan)
    # For simplicity, if this function isn't the performance bottleneck,
    # the standard NumPy approach often still works in JAX.

    # If using JAX's functional scan:
    def scan_fn(carry, x):
        (reward,) = x
        new_ret = reward + gamma * carry
        return new_ret, new_ret

    # rewards are (r_t, r_{t+1}, ..., r_{T-1})
    # reverse rewards: (r_{T-1}, ..., r_t)
    # rewards are padded to include the bootstrap value for the scan start
    padded_rewards = jnp.concatenate([rewards, jnp.array([0.0])])

    # The scan computes: R_{t-1} = r_{t-1} + gamma * R_t
    _, returns_scanned = lax.scan(
        scan_fn,
        bootstrap,  # initial carry R_T = bootstrap
        padded_rewards[:-1][::-1],  # scan over (r_{T-1}, r_{T-2}, ..., r_0)
        reverse=False,  # scan from T-1 down to 0
    )

    # The result returns_scanned is (R_{T-1}, R_{T-2}, ..., R_0). Reverse it back.
    # Note: We discard the last step's reward (r_{T-1}) from the scan input because the
    # bootstrap covers the V(s_T). The scan naturally produces R_0 to R_{T-1}.
    return returns_scanned[::-1]


# ---------------------------------------------------------------------------
# 2. GAE (Generalized Advantage Estimation) (JAX)
# ---------------------------------------------------------------------------


def compute_gae_jax(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    gamma: float,
    lam: float,
    bootstrap_values: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute GAE over a *sequence of episodes* contained in a single rollout.
    (JAX implementation using jax.lax.scan for efficiency)
    """
    # T = len(rewards)
    # advantages = jnp.zeros_like(rewards, dtype=jnp.float32)

    # This loop over episodes is generally discouraged in JAX for compilation.
    # The most JAX-idiomatic GAE computation for a single contiguous buffer:

    # 1. Calculate TD-Error (deltas)
    # We need V(s_{t+1}) for t=0 to T-1. V(s_T) is the final value.
    # The values array is V(s_0) to V(s_{T-1}).

    # Determine the next value V(s_{t+1})
    # If `dones[t]` is True, V(s_{t+1}) should be 0, otherwise it's `values[t+1]`
    next_values = jnp.concatenate([values[1:], bootstrap_values])

    # Apply mask for terminal states: V(s_{t+1}) = V(s_{t+1}) * (1 - dones_mask)
    # Since dones here marks terminal states, we use it to zero out V(s_{t+1}).
    # Assuming 'dones' is 1.0 for terminated/truncated, 0.0 otherwise.
    dones_mask = dones.astype(jnp.float32)
    next_values_masked = next_values * (1.0 - dones_mask * gamma)

    # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    deltas = rewards + gamma * next_values_masked - values

    # 2. Generalized Advantage Estimation (GAE) via scan
    # Advantage_t = delta_t + (gamma*lam) * Advantage_{t+1} * (1 - dones_mask)

    # We scan backwards. We need (gamma*lam) * (1 - dones_mask) which is the discount factor.
    # GAE factor is gamma*lam, but it becomes 0 at a terminal step.
    # JAX's lax.scan is excellent for this recurrence.
    gae_factor = gamma * lam

    # The recurrence relation:
    # A_t = delta_t + gae_factor * A_{t+1} * (1 - dones[t])

    # The scan function: carry is A_{t+1}, input is (delta_t, dones_t)
    def gae_scan_fn(gae_next, inputs):
        delta_t, done_t = inputs
        gae_t = delta_t + gae_factor * gae_next * (1.0 - done_t)
        return gae_t, gae_t  # next carry is gae_t, output is gae_t

    # Scan inputs: reversed deltas and reversed dones
    # The bootstrap values array here is effectively V(s_T), which is handled
    # by the delta calculation above. The GAE scan starts with A_T = 0.

    # Note: `bootstrap_values` is an array of value estimates *at episode boundaries*.
    # The original implementation iterates over episodes, which is less ideal for JAX.
    # For a fully vectorized JAX version, we rely on the `dones` mask.
    # Assuming the input `bootstrap_values` has already been incorporated into the `next_values`
    # or that the `dones` array correctly captures episode ends.
    # We will assume a single continuous rollout (which PPO often does) and rely on the masking.

    # Initial carry: A_{T-1}
    # Scan starts at t=T-1 and goes backwards to t=0.

    # Perform reverse scan for GAE
    initial_gae = jnp.array(0.0)  # A_T = 0
    _, gae_advantages = lax.scan(
        gae_scan_fn,
        initial_gae,
        (deltas[::-1], dones_mask[::-1]),  # Reversed deltas and dones
        reverse=False,
    )

    # Reverse back to original order
    return gae_advantages[::-1]


# ---------------------------------------------------------------------------
# 3. Advantage normalization (JAX)
# ---------------------------------------------------------------------------


def standardize_advantages_jax(advantages: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """
    Normalize advantages to have zero mean and unit variance. (JAX implementation)
    """
    # jnp.mean and jnp.std are JAX equivalents
    mean = jnp.mean(advantages)
    std = jnp.std(advantages)
    return (advantages - mean) / (std + eps)
