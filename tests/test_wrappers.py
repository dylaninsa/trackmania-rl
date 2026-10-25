import subprocess
import sys

import gymnasium
import numpy as np
import pytest

from envs.wrappers import ACTIONS, LAST_LIDAR, LIDAR, OBS_DIM, SPEED, FlattenLidarObservation


class FakeLidarEnv(gymnasium.Env):
    """Mimics TMRL's LIDAR observation space without the game.

    Args:
        lidar_shape (tuple[int, int]): Shape of the LIDAR history.
        nb_actions (int): Number of past actions in the observation.
    """

    def __init__(self, lidar_shape=(4, 19), nb_actions=2):
        act = gymnasium.spaces.Box(-1.0, 1.0, shape=(3,))
        self.observation_space = gymnasium.spaces.Tuple((
            gymnasium.spaces.Box(0.0, 1000.0, shape=(1,)),
            gymnasium.spaces.Box(0.0, np.inf, shape=lidar_shape),
            *(act,) * nb_actions,
        ))
        self.action_space = act
        self.lidar = np.arange(np.prod(lidar_shape), dtype=np.float32).reshape(lidar_shape)

    def _obs(self):
        return (np.array([250.0], dtype=np.float32), self.lidar,
                np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, -0.6]))

    def reset(self, seed=None, options=None):
        return self._obs(), {}

    def step(self, action):
        return self._obs(), 0.0, False, False, {}


def test_shape_dtype_and_space():
    env = FlattenLidarObservation(FakeLidarEnv())
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,) == (83,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)


def test_layout_and_scaling():
    fake = FakeLidarEnv()
    env = FlattenLidarObservation(fake, speed_scale=500.0, lidar_scale=100.0)
    obs, _ = env.reset()
    np.testing.assert_allclose(obs[SPEED], [0.5])
    np.testing.assert_allclose(obs[LIDAR], fake.lidar.ravel() / 100.0)
    np.testing.assert_allclose(obs[LAST_LIDAR], fake.lidar[-1] / 100.0)
    np.testing.assert_allclose(obs[ACTIONS], [0.1, 0.2, 0.3, 0.4, 0.5, -0.6], rtol=1e-6)


def test_step_is_flattened_too():
    env = FlattenLidarObservation(FakeLidarEnv())
    env.reset()
    obs, *_ = env.step(env.action_space.sample())
    assert obs.shape == (OBS_DIM,)


@pytest.mark.parametrize("kwargs", [{"lidar_shape": (1, 19)}, {"nb_actions": 1}])
def test_rejects_unexpected_observation(kwargs):
    with pytest.raises(ValueError):
        FlattenLidarObservation(FakeLidarEnv(**kwargs))


def test_import_does_not_load_tmrl():
    code = "import sys, envs.trackmania, envs.wrappers; assert 'tmrl' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
