import numpy as np

from nautilus.core.on_policy import compute_gae, standardize_advantages


def test_gae_simple_discount():
    """
    Verify GAE logic on a simple 3-step sequence with Gamma=0.5, Lambda=1.0
    Formula: Adv_t = r_t + gamma * V_next - V_t + (gamma * lam * Adv_next)
    """
    # Setup: 3 steps, 1 environment
    # Rewards: [1, 1, 1]
    # Values:  [0, 0, 0] (To make math easy, V(s) is 0, so Advantage = Q)
    # Gamma: 0.5
    rewards = np.array([[1.0], [1.0], [1.0]])  # Shape (T, N)
    values = np.zeros((3, 1))
    dones = np.zeros((3, 1))
    next_val = np.array([0.0])
    next_done = np.array([0.0])

    # Expected Calculation (Backwards):
    # T=2: r=1 + 0.5*0 - 0 = 1.0
    # T=1: r=1 + 0.5*1.0 = 1.5
    # T=0: r=1 + 0.5*1.5 = 1.75

    adv = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        gamma=0.5,
        lam=1.0,
        next_value=next_val,
        next_done=next_done,
    )

    expected = np.array([[1.75], [1.5], [1.0]])
    np.testing.assert_allclose(adv, expected, atol=1e-5)


def test_gae_with_done():
    """
    Verify that a 'Done' flag correctly cuts off bootstrapping.
    """
    # Setup: 3 steps. Step 1 (middle) is terminal.
    rewards = np.array([[1.0], [1.0], [1.0]])
    values = np.zeros((3, 1))
    dones = np.array([[0.0], [1.0], [0.0]])  # Second step is terminal

    # Gamma 0.9
    adv = compute_gae(
        rewards,
        values,
        dones,
        gamma=0.9,
        lam=1.0,
        next_value=np.array([0.0]),
        next_done=np.array([0.0]),
    )

    # T=2: 1.0 (Standard)
    # T=1: 1.0 (Terminal! Should NOT look at T=2. Adv = r + 0 - V)
    # T=0: 1.0 + 0.9 * 1.0 = 1.9

    expected = np.array([[1.9], [1.0], [1.0]])
    np.testing.assert_allclose(adv, expected, atol=1e-5)


def test_standardization():
    data = np.array([1.0, 2.0, 3.0])
    std_data = standardize_advantages(data)
    assert abs(std_data.mean()) < 1e-5
    assert abs(std_data.std() - 1.0) < 1e-5


def test_standardization_constant_returns_zero():
    data = np.ones(5)
    std_data = standardize_advantages(data)
    assert np.allclose(std_data, np.zeros_like(data))


def test_gae_mixed_dones_vectorized():
    """
    Validate that bootstrapping stops per-env when dones differ across the batch.
    """
    # Two environments, three timesteps
    rewards = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    values = np.zeros_like(rewards)
    # Env0 terminates at t=1, Env1 is still alive
    dones = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])

    adv = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        gamma=0.9,
        lam=1.0,
        next_value=np.array([0.0, 0.0]),
        next_done=np.array([0.0, 0.0]),
    )

    # Env0: t2=1, t1 terminal ->1, t0=1+0.9*1=1.9
    # Env1: t2=3, t1=2+0.9*3=4.7, t0=1+0.9*4.7=5.23
    expected = np.array([[1.9, 5.23], [1.0, 4.7], [1.0, 3.0]])
    np.testing.assert_allclose(adv, expected, atol=1e-4)
