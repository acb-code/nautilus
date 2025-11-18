import pytest


def test_lunarlander_time_limit_matches_gymnasium_spec():
    gym = pytest.importorskip("gymnasium")

    try:
        spec = gym.spec("LunarLander-v3")
    except Exception as exc:  # pragma: no cover - depends on optional Box2D install
        pytest.skip(f"LunarLander-v3 not available: {exc}")

    # Gymnasium registers LunarLander with a 1000-step limit and a 60-second wall-clock cap.
    assert spec.max_episode_steps == 1000
    assert spec.max_episode_seconds == 60
