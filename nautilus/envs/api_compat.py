from typing import Any


def reset_env(env) -> tuple[Any, dict]:
    """
    Reset an environment and return (obs, info), handling both Gym and Gymnasium APIs.
    """
    outputs = env.reset()
    # Gymnasium: (obs, info)
    if isinstance(outputs, tuple):
        return outputs
    # Old Gym: obs only
    else:
        return outputs, {}


def step_env(env, action, truncated: bool = False):
    """
    Step an environment and return (next_obs, reward, terminated, truncated, info),
    handling both Gym and Gymnasium APIs.

    The `truncated` flag is passed in so that algorithms that enforce their own
    max episode length can override the termination condition if needed.
    """
    step_output = env.step(action)

    # Old Gym: (obs, reward, done, info)
    if len(step_output) == 4:
        next_obs, reward, done, info = step_output
        terminated = done
        # If the caller already decided the episode is truncated
        if truncated:
            terminated = False
        return next_obs, reward, terminated, truncated, info

    # Gymnasium: (obs, reward, terminated, truncated, info)
    else:
        next_obs, reward, terminated, truncated_env, info = step_output

        # Respect caller's truncation if they passed one in
        if truncated:
            terminated = False
            truncated_final = True
        else:
            truncated_final = truncated_env

        return next_obs, reward, terminated, truncated_final, info
