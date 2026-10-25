import pickle

import numpy as np
import pytest

try:
    import tmrl.custom.tm.tm_gym_interfaces as tm_interfaces
    from envs.tm_interface import LidarInterface
except (ImportError, RuntimeError) as error:
    pytest.skip(f"TMRL is not usable on this machine: {error}", allow_module_level=True)


def test_initialize_uses_our_trajectory(tmp_path, monkeypatch):
    trajectory = np.stack([np.arange(50) * 0.1, np.zeros(50), np.zeros(50)], axis=1)
    reward_path = tmp_path / "trajectory.pkl"
    with open(reward_path, "wb") as f:
        pickle.dump(trajectory, f)

    def fake_initialize_common(self):
        self.window_interface = type("FakeWindow", (), {"screenshot": lambda _: None})()
        self.reward_function = None

    monkeypatch.setattr(tm_interfaces.TM2020Interface, "initialize_common", fake_initialize_common)
    monkeypatch.setattr(tm_interfaces, "Lidar", lambda img: None)

    interface = LidarInterface(reward_path=reward_path)
    interface.initialize()

    np.testing.assert_array_equal(interface.reward_function.data, trajectory)
