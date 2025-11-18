import numpy as np

from nautilus.core.on_policy import OnPolicyBuffer


def test_on_policy_buffer_stacks_arrays_and_exposes_values():
    buffer = OnPolicyBuffer(rollout_length=3, num_envs=1)

    for idx in range(3):
        buffer.add(
            obs=np.array([idx], dtype=np.float32),
            action=np.array([idx + 1]),
            reward=np.array([1.0]),
            done=np.array([0.0]),
            info={"val": np.array([idx], dtype=np.float32), "log_prob": np.array([-0.5])},
        )

    batch = buffer.get()

    assert batch["obs"].shape == (3, 1)
    assert batch["actions"].shape == (3, 1)
    np.testing.assert_array_equal(batch["values"][:, 0], np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(batch["log_probs"].reshape(-1), np.array([-0.5, -0.5, -0.5]))


def test_on_policy_buffer_computes_returns_and_advantages():
    buffer = OnPolicyBuffer(rollout_length=2, num_envs=1)

    buffer.add(
        obs=np.array([0.0]),
        action=np.array([0]),
        reward=np.array([1.0]),
        done=np.array([0.0]),
        info={"val": np.array([0.0]), "log_prob": np.array([-0.1])},
    )
    buffer.add(
        obs=np.array([1.0]),
        action=np.array([1]),
        reward=np.array([1.0]),
        done=np.array([0.0]),
        info={"val": np.array([0.0]), "log_prob": np.array([-0.2])},
    )

    advantages, returns = buffer.compute_returns_and_advantages(
        last_value=np.array([0.0]),
        last_done=np.array([0.0]),
        gamma=1.0,
        lam=1.0,
        normalize_advantages=True,
    )

    # With rewards of 1 and values=0, raw advantages are [2, 1].
    np.testing.assert_array_equal(returns[:, 0], np.array([2.0, 1.0]))
    assert advantages.shape == (2, 1)
    assert np.isclose(advantages.mean(), 0.0, atol=1e-6)
    assert buffer.advantages is not None
    assert buffer.returns is not None
