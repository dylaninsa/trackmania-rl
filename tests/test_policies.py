import numpy as np
import pytest

from baselines.policies import PIDPolicy, RandomPolicy
from envs.trackmania import LAST_LIDAR, SPEED, SPEED_SCALE

RIGHT_OPEN = np.array([0.6] * 9 + [0.5] + [0.2] * 9)


def make_obs(lidar, speed_mps):
    """Builds an observation of size 83 with the given last LIDAR scan and speed.

    Args:
        lidar (np.ndarray): The 19 normalized distances of the most recent scan.
        speed_mps (float): Speed in m/s.

    Returns:
        np.ndarray: Float32 observation.
    """
    obs = np.zeros(83, dtype=np.float32)
    obs[SPEED] = speed_mps / SPEED_SCALE
    obs[LAST_LIDAR] = lidar
    return obs


def test_random_action_shape_and_bounds():
    action = RandomPolicy(seed=0).act(None)

    assert action.shape == (3,)
    assert action.dtype == np.float32
    assert np.all((action >= -1.0) & (action <= 1.0))


def test_random_is_reproducible_with_a_seed():
    a, b = RandomPolicy(seed=0), RandomPolicy(seed=0)

    for _ in range(5):
        np.testing.assert_array_equal(a.act(None), b.act(None))


def test_steer_towards_the_open_side():
    assert PIDPolicy(kp=1.0, ki=0.0, kd=0.0).steer(RIGHT_OPEN) == pytest.approx(0.4)
    assert PIDPolicy(kp=1.0, ki=0.0, kd=0.0).steer(RIGHT_OPEN[::-1]) == pytest.approx(-0.4)


def test_steer_is_zero_when_centered():
    assert PIDPolicy().steer(np.full(19, 0.5)) == 0.0


def test_steer_is_clipped():
    assert PIDPolicy(kp=10.0, ki=0.0, kd=0.0).steer(RIGHT_OPEN) == 1.0


def test_steer_derivative_is_zero_on_first_step():
    pid = PIDPolicy(kp=0.0, ki=0.0, kd=0.1)

    assert pid.steer(RIGHT_OPEN) == 0.0
    assert pid.steer(np.array([0.4] * 9 + [0.5] + [0.2] * 9)) == pytest.approx(-0.4)


def test_reset_clears_the_pid_state():
    pid = PIDPolicy(ki=1.0)
    pid.steer(RIGHT_OPEN)

    pid.reset()

    assert (pid.integral, pid.previous_error) == (0.0, None)


@pytest.mark.parametrize("speed, expected", [(20.0, (1.0, 0.0)), (37.0, (0.0, 0.0)), (45.0, (0.0, 1.0))])
def test_throttle_around_the_target_speed(speed, expected):
    assert PIDPolicy().throttle(np.full(19, 0.5), speed, steer=0.0) == expected


def test_throttle_slows_down_when_steering():
    pid = PIDPolicy()

    assert pid.throttle(np.full(19, 0.5), 30.0, steer=0.0) == (1.0, 0.0)
    assert pid.throttle(np.full(19, 0.5), 30.0, steer=0.8) == (0.0, 1.0)


def test_throttle_never_targets_below_min_speed():
    assert PIDPolicy(min_speed=10.0).throttle(np.full(19, 0.5), 8.0, steer=1.0) == (1.0, 0.0)


def test_pid_act_combines_steer_and_throttle():
    action = PIDPolicy(kp=1.0, ki=0.0, kd=0.0).act(make_obs(RIGHT_OPEN, 20.0))

    assert action.dtype == np.float32
    np.testing.assert_allclose(action, [1.0, 0.0, 0.4], rtol=1e-6)
