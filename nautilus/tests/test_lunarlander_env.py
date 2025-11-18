import pytest


def test_lunarlander_time_limit_matches_gymnasium_spec():
    gym = pytest.importorskip("gymnasium")

    try:
        spec = gym.spec("LunarLander-v3")
    except Exception as exc:  # pragma: no cover - depends on optional Box2D install
        pytest.skip(f"LunarLander-v3 not available: {exc}")

    # Gymnasium registers LunarLander with a 1000-step limit. Some versions also
    # attach a 60-second wall-clock cap; others omit the seconds attribute entirely.
    assert spec.max_episode_steps == 1000
    max_seconds = getattr(spec, "max_episode_seconds", None)
    assert max_seconds in (None, 60)
