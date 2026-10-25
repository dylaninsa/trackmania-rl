import pickle

import numpy as np
import pytest

from envs.reward import TrajectoryReward, load_trajectory


def make_line(n=100):
    """Builds a straight trajectory along the x axis with points spaced 0.1 m apart.

    Args:
        n (int): Number of points.

    Returns:
        np.ndarray: Trajectory of shape ``(n, 3)``.
    """
    return np.stack([np.arange(n) * 0.1, np.zeros(n), np.zeros(n)], axis=1)


def test_load_trajectory(tmp_path):
    path = tmp_path / "trajectory.pkl"
    path.write_bytes(pickle.dumps(make_line()))

    np.testing.assert_array_equal(load_trajectory(path), make_line())


def test_reward_counts_points_passed():
    rf = TrajectoryReward(make_line())

    assert rf.compute_reward(np.array([1.0, 0.0, 0.0])) == (pytest.approx(0.1), False)
    assert rf.compute_reward(np.array([2.0, 0.0, 0.0])) == (pytest.approx(0.1), False)


def test_no_reward_for_moving_backwards():
    rf = TrajectoryReward(make_line())
    rf.compute_reward(np.array([2.0, 0.0, 0.0]))

    reward, _ = rf.compute_reward(np.array([1.0, 0.0, 0.0]))

    assert reward == 0.0
    assert rf.index == 20


def test_search_is_limited_ahead():
    rf = TrajectoryReward(make_line(), max_forward_points=50)

    rf.compute_reward(np.array([9.0, 0.0, 0.0]))

    assert rf.index == 49


def test_terminates_when_stuck():
    rf = TrajectoryReward(make_line(), patience=3, grace_steps=2)
    rf.compute_reward(np.array([1.0, 0.0, 0.0]))
    rf.compute_reward(np.array([2.0, 0.0, 0.0]))

    terminated = [rf.compute_reward(np.array([2.0, 0.0, 0.0]))[1] for _ in range(3)]

    assert terminated == [False, False, True]


def test_no_stuck_count_during_grace_steps():
    rf = TrajectoryReward(make_line(), grace_steps=5)

    for _ in range(5):
        rf.compute_reward(np.array([0.0, 0.0, 0.0]))

    assert rf.steps_without_progress == 0


def test_terminates_when_off_track():
    rf = TrajectoryReward(make_line(), max_distance_m=5.0)

    assert rf.compute_reward(np.array([1.0, 6.0, 0.0])) == (0.0, True)
    assert rf.index == 0
    assert rf.dist_to_traj == pytest.approx(6.0)


def test_progress_and_distance():
    rf = TrajectoryReward(make_line())
    assert (rf.progress, rf.distance_m) == (0.0, 0.0)

    rf.index = 99

    assert rf.progress == 1.0
    assert rf.distance_m == pytest.approx(9.9)


def test_reset():
    rf = TrajectoryReward(make_line(), grace_steps=0)
    rf.compute_reward(np.array([2.0, 0.0, 0.0]))
    rf.compute_reward(np.array([2.0, 1.0, 0.0]))

    rf.reset()

    assert (rf.index, rf.steps, rf.steps_without_progress, rf.dist_to_traj) == (0, 0, 0, 0.0)
