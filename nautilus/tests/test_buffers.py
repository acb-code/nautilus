import numpy as np
from nautilus.core.buffers import ReplayBuffer

def test_replay_buffer_shapes():
    rb = ReplayBuffer(10, (4,), (1,))
    for i in range(7):
        rb.add(np.ones(4)*i, [0], 1.0, np.ones(4)*(i+1), False)
    obs, act, rew, nxt, done = rb.sample(5)
    assert obs.shape == (5,4)
    assert act.shape == (5,1)
    assert nxt.shape == (5,4)
    assert rew.shape == (5,)
    assert done.shape == (5,)
