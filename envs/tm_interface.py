"""Interface rtgym pour TrackMania 2020 en mode LIDAR, avec notre récompense et un `info` rempli.

Ce module importe TMRL : ne l'importer que sur une machine où TMRL et TmrlData sont installés.
"""

import numpy as np
import tmrl.config.config_constants as cfg
from tmrl.custom.tm.tm_gym_interfaces import TM2020InterfaceLidar
from tmrl.custom.tm.utils.compute_reward import RewardFunction

POINT_SPACING_M = 0.1  # espacement des points de la trajectoire enregistrée par `tmrl --record-reward`


class LidarInterface(TM2020InterfaceLidar):
    def __init__(self, reward_path, img_hist_len=4, gamepad=True):
        super().__init__(img_hist_len=img_hist_len, gamepad=gamepad)
        self.reward_path = str(reward_path)

    def initialize_common(self):
        super().initialize_common()
        self.reward_function = self.make_reward_function()

    def make_reward_function(self):
        # Provisoire : récompense de TMRL, à remplacer par envs/reward.py (étape 5).
        rc = cfg.REWARD_CONFIG
        return RewardFunction(reward_data_path=self.reward_path,
                              nb_obs_forward=rc["CHECK_FORWARD"],
                              nb_obs_backward=rc["CHECK_BACKWARD"],
                              nb_zero_rew_before_failure=rc["FAILURE_COUNTDOWN"],
                              min_nb_steps_before_failure=rc["MIN_STEPS"],
                              max_dist_from_traj=rc["MAX_STRAY"])

    def send_control(self, control):
        # La manette virtuelle refuse toute valeur hors de [-1, 1] (assert dans control_gamepad).
        if control is not None:
            control = np.clip(control, -1.0, 1.0)
        super().send_control(control)

    def reset(self, seed=None, options=None):
        self.reset_common()
        lidar, speed, data = self.grab_lidar_speed_and_data()
        for _ in range(self.img_hist_len):
            self.img_hist.append(lidar)
        self.reward_function.reset()
        return [speed, self.lidar_history()], self.make_info(data)

    def get_obs_rew_terminated_info(self):
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
        return np.array(list(self.img_hist), dtype=np.float32)  # (img_hist_len, 19), le plus récent en dernier

    def make_info(self, data):
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
