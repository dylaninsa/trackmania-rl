import pickle

import gymnasium
import numpy as np
import tmrl.custom.tm.tm_gym_interfaces as tm_interfaces

from envs.trackmania import (ACTIONS, LAST_LIDAR, LIDAR, LIDAR_SCALE, SPEED, SPEED_SCALE, FlattenLidarObservation,
                             LidarInterface)


class FakeLidarEnv(gymnasium.Env):
    """Returns a fixed observation shaped like TMRL's LIDAR observation."""

    observation_space = gymnasium.spaces.Tuple(())
    action_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(3,))
    lidar = np.arange(76, dtype=np.float32).reshape(4, 19)

    def reset(self, seed=None, options=None):
        return (np.array([50.0], dtype=np.float32), self.lidar, np.array([0.1, 0.2, 0.3]),
                np.array([0.4, 0.5, -0.6])), {}


def test_flattened_observation_layout():
    obs, _ = FlattenLidarObservation(FakeLidarEnv()).reset()

    assert obs.shape == (83,)
    assert obs.dtype == np.float32
    np.testing.assert_allclose(obs[SPEED], [50.0 / SPEED_SCALE])
    np.testing.assert_allclose(obs[LIDAR], FakeLidarEnv.lidar.ravel() / LIDAR_SCALE)
    np.testing.assert_allclose(obs[LAST_LIDAR], FakeLidarEnv.lidar[-1] / LIDAR_SCALE)
    np.testing.assert_allclose(obs[ACTIONS], [0.1, 0.2, 0.3, 0.4, 0.5, -0.6], rtol=1e-6)


def test_interface_reads_our_trajectory(tmp_path, monkeypatch):
    trajectory = np.stack([np.arange(50) * 0.1, np.zeros(50), np.zeros(50)], axis=1)
    reward_path = tmp_path / "trajectory.pkl"
    reward_path.write_bytes(pickle.dumps(trajectory))

    def fake_initialize_common(self):
        self.window_interface = type("FakeWindow", (), {"screenshot": lambda _: None})()

    monkeypatch.setattr(tm_interfaces.TM2020Interface, "initialize_common", fake_initialize_common)
    monkeypatch.setattr(tm_interfaces, "Lidar", lambda img: None)

    interface = LidarInterface(reward_path)
    interface.initialize()

    np.testing.assert_array_equal(interface.reward_function.data, trajectory)
