import numpy as np
import torch

from algos.replay_buffer import ReplayBuffer


def _fill(buf, n, obs_dim=4, act_dim=2, start=0):
    for k in range(start, start + n):
        buf.add(np.full(obs_dim, k), np.full(act_dim, k), float(k), np.full(obs_dim, k + 1), k % 2 == 0)


def test_len_grows_then_saturates():
    buf = ReplayBuffer(4, 2, capacity=5)
    assert len(buf) == 0
    _fill(buf, 3)
    assert len(buf) == 3
    _fill(buf, 10, start=3)
    assert len(buf) == 5


def test_circular_overwrite_keeps_latest():
    buf = ReplayBuffer(4, 2, capacity=5)
    _fill(buf, 8)  # transitions 3..7 must remain
    batch = buf.sample(200)
    assert set(batch["rew"].tolist()) <= {3.0, 4.0, 5.0, 6.0, 7.0}


def test_sample_shapes_dtypes():
    buf = ReplayBuffer(4, 2, capacity=10)
    _fill(buf, 10)
    b = buf.sample(6)
    assert b["obs"].shape == (6, 4)
    assert b["next_obs"].shape == (6, 4)
    assert b["act"].shape == (6, 2)
    assert b["rew"].shape == (6,) 
    assert b["terminated"].shape == (6,)
    assert all(t.dtype == torch.float32 for t in b.values())


def test_transition_consistency_and_terminated_flag():
    buf = ReplayBuffer(4, 2, capacity=10)
    _fill(buf, 10)
    b = buf.sample(50)
    k = b["rew"]
    assert torch.equal(b["obs"][:, 0], k)
    assert torch.equal(b["next_obs"][:, 0], k + 1)
    assert torch.equal(b["terminated"], (k % 2 == 0).float())


def test_sample_empty_raises():
    buf = ReplayBuffer(4, 2, capacity=10)
    try:
        buf.sample(1)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_same_seed_same_samples():
    a, b = ReplayBuffer(4, 2, capacity=10, seed=7), ReplayBuffer(4, 2, capacity=10, seed=7)
    _fill(a, 10)
    _fill(b, 10)
    assert torch.equal(a.sample(8)["rew"], b.sample(8)["rew"])