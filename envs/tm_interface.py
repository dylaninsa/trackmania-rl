"""rtgym interface for TrackMania 2020 in LIDAR mode, with our reward and a filled ``info`` dictionary.

This module imports TMRL: only import it on a machine where TMRL and the TmrlData folder are installed.

Attributes:
    POINT_SPACING_M (float): Distance in meters between two points of a trajectory recorded with
        ``python -m tmrl --record-reward``.
"""

import numpy as np
import tmrl.config.config_constants as cfg
from tmrl.custom.tm.tm_gym_interfaces import TM2020InterfaceLidar
from tmrl.custom.tm.utils.compute_reward import RewardFunction

POINT_SPACING_M = 0.1


class LidarInterface(TM2020InterfaceLidar):
    """TMRL LIDAR interface that uses our reward function and reports progress in ``info``.

    Args:
        reward_path (str | pathlib.Path): Demonstration trajectory (.pkl) of the track loaded in the game.
        img_hist_len (int): Number of LIDAR scans kept in the observation.
        gamepad (bool): Whether to control the car with the virtual gamepad (analog actions).
    """

    def __init__(self, reward_path, img_hist_len=4, gamepad=True):
        super().__init__(img_hist_len=img_hist_len, gamepad=gamepad)
        self.reward_path = str(reward_path)

    def initialize(self):
        """Initializes the game connection, then replaces TMRL's reward function with ours.

        ``initialize`` is overridden rather than ``initialize_common`` because TMRL's LIDAR interface calls
        ``super().initialize_common()``, which would bypass an override of ``initialize_common``.
        """
        super().initialize()
        self.reward_function = self.make_reward_function()

    def make_reward_function(self):
        """Builds the reward function.

        Provisional: uses TMRL's reward function until ``envs/reward.py`` replaces it.

        Returns:
            RewardFunction: Object exposing ``compute_reward(pos)``, ``reset()``, ``cur_idx``, ``data``
                and ``datalen``.
        """
        rc = cfg.REWARD_CONFIG
        return RewardFunction(reward_data_path=self.reward_path,
                              nb_obs_forward=rc["CHECK_FORWARD"],
                              nb_obs_backward=rc["CHECK_BACKWARD"],
                              nb_zero_rew_before_failure=rc["FAILURE_COUNTDOWN"],
                              min_nb_steps_before_failure=rc["MIN_STEPS"],
                              max_dist_from_traj=rc["MAX_STRAY"])

    def send_control(self, control):
        """Sends an action to the game, clipped to [-1, 1] because the virtual gamepad rejects other values.

        Args:
            control (np.ndarray | None): ``[gas, brake, steer]``, or None to send nothing.
        """
        if control is not None:
            control = np.clip(control, -1.0, 1.0)
        super().send_control(control)

    def reset(self, seed=None, options=None):
        """Restarts the race and returns the first observation.

        Args:
            seed (int | None): Unused, the game is not seedable.
            options (dict | None): Unused.

        Returns:
            tuple[list, dict]: Observation ``[speed, lidar_history]`` and the ``info`` dictionary.
        """
        self.reset_common()
        lidar, speed, data = self.grab_lidar_speed_and_data()
        for _ in range(self.img_hist_len):
            self.img_hist.append(lidar)
        self.reward_function.reset()
        return [speed, self.lidar_history()], self.make_info(data)

    def get_obs_rew_terminated_info(self):
        """Captures the current observation and computes the reward.

        Returns:
            tuple[list, np.float32, bool, dict]: Observation ``[speed, lidar_history]``, reward, terminated
                flag and ``info`` dictionary.
        """
        lidar, speed, data = self.grab_lidar_speed_and_data()
        rew, terminated = self.reward_function.compute_reward(pos=np.array([data[2], data[3], data[4]]))
        self.img_hist.append(lidar)
        info = self.make_info(data)
        if info["end_of_track"]:
            rew += self.finish_reward
            terminated = True
        rew += self.constant_penalty
        return [speed, self.lidar_history()], np.float32(rew), terminated, info

    def lidar_history(self):
        """Returns the LIDAR history.

        Returns:
            np.ndarray: Array of shape ``(img_hist_len, 19)``, most recent scan last.
        """
        return np.array(list(self.img_hist), dtype=np.float32)

    def make_info(self, data):
        """Builds the ``info`` dictionary from the OpenPlanet data.

        Args:
            data (tuple): Values sent by the TMRL OpenPlanet plugin.

        Returns:
            dict: ``pos``, ``speed``, ``end_of_track``, ``traj_index``, ``progress`` (0 to 1),
                ``distance_m`` and ``dist_to_traj``.
        """
        rf = self.reward_function
        pos = np.array([data[2], data[3], data[4]], dtype=np.float32)
        idx = rf.cur_idx
        return {
            "pos": pos,
            "speed": float(data[0]),
            "end_of_track": bool(data[8]),
            "traj_index": int(idx),
            "progress": idx / max(rf.datalen - 1, 1),
            "distance_m": idx * POINT_SPACING_M,
            "dist_to_traj": float(np.linalg.norm(pos - rf.data[idx])),
        }
