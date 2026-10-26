import pickle
from collections import deque

import gymnasium
import numpy as np
import pytest
import tmrl.custom.tm.tm_gym_interfaces as tm_interfaces

from envs.reward import TrajectoryReward
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

    np.testing.assert_array_equal(interface.reward_function.trajectory, trajectory)


def test_interface_step_gives_our_finish_bonus(monkeypatch):
    lidar = np.ones(19, dtype=np.float32)
    speed = np.array([30.0], dtype=np.float32)
    data = (30.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    monkeypatch.setattr(tm_interfaces.TM2020InterfaceLidar, "grab_lidar_speed_and_data",
                        lambda self: (lidar, speed, data))
    line = np.stack([np.arange(100) * 0.1, np.zeros(100), np.zeros(100)], axis=1)
    interface = LidarInterface("unused.pkl")
    interface.reward_function = TrajectoryReward(line, finish_bonus=100.0)
    interface.img_hist = deque([np.zeros(19, dtype=np.float32)] * 4, maxlen=4)

    obs, reward, terminated, info = interface.get_obs_rew_terminated_info()

    assert len(obs) == 2
    assert obs[1].shape == (4, 19)
    np.testing.assert_array_equal(obs[1][-1], lidar)
    assert reward == pytest.approx(100.1)
    assert terminated
    assert info["end_of_track"]
    assert info["distance_m"] == pytest.approx(1.0)
