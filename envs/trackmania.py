"""TrackMania 2020 environment in LIDAR mode, used by the whole project.

The observation is a float32 vector of size 83:

* index 0: speed in m/s, divided by ``SPEED_SCALE``;
* indices 1 to 76: 4 LIDAR scans of 19 beams, oldest first, divided by ``LIDAR_SCALE``. In a scan, beam 0
  points right, beam 9 straight ahead and beam 18 left;
* indices 77 to 82: the 2 previous actions, oldest first.

An action is ``[gas, brake, steer]`` in [-1, 1]. Only positive gas and brake values have an effect, and a
positive steer turns right. A step lasts 0.05 s.

Attributes:
    SPEED (slice): Speed in the observation.
    LIDAR (slice): The 4 LIDAR scans in the observation.
    LAST_LIDAR (slice): The most recent LIDAR scan in the observation.
    ACTIONS (slice): The 2 previous actions in the observation.
    SPEED_SCALE (float): Speed normalization constant. The speed peaks around 80 m/s on tmrl-test.
    LIDAR_SCALE (float): LIDAR normalization constant. The longest beam observed on tmrl-test is 483 px.
    POINT_SPACING_M (float): Distance in meters between two points of a recorded trajectory.
    DEFAULT_REWARD_PATH (pathlib.Path): Demonstration trajectory of the tmrl-test track.
"""

from pathlib import Path

import gymnasium
import numpy as np
import tmrl.config.config_constants as cfg
from tmrl.config.config_objects import CONFIG_DICT
from tmrl.custom.tm.tm_gym_interfaces import TM2020InterfaceLidar
from tmrl.custom.tm.utils.compute_reward import RewardFunction

SPEED = slice(0, 1)
LIDAR = slice(1, 77)
LAST_LIDAR = slice(58, 77)
ACTIONS = slice(77, 83)

SPEED_SCALE = 100.0
LIDAR_SCALE = 500.0
POINT_SPACING_M = 0.1
DEFAULT_REWARD_PATH = Path(__file__).resolve().parents[1] / "data" / "reward" / "tmrl-test.pkl"


class LidarInterface(TM2020InterfaceLidar):
    """TMRL LIDAR interface that reads our trajectory and reports progress in ``info``.

    Args:
        reward_path (str | pathlib.Path): Demonstration trajectory (.pkl) of the track loaded in the game.
    """

    def __init__(self, reward_path):
        super().__init__(img_hist_len=4, gamepad=True)
        self.reward_path = str(reward_path)
        self.data = None

    def initialize(self):
        """Connects to the game, then replaces TMRL's reward function with one that reads our trajectory.

        The reward computation itself is still TMRL's, until ``envs/reward.py`` replaces it.
        """
        super().initialize()
        rc = cfg.REWARD_CONFIG
        self.reward_function = RewardFunction(reward_data_path=self.reward_path,
                                              nb_obs_forward=rc["CHECK_FORWARD"],
                                              nb_obs_backward=rc["CHECK_BACKWARD"],
                                              nb_zero_rew_before_failure=rc["FAILURE_COUNTDOWN"],
                                              min_nb_steps_before_failure=rc["MIN_STEPS"],
                                              max_dist_from_traj=rc["MAX_STRAY"])

    def send_control(self, control):
        """Sends an action clipped to [-1, 1], because the virtual gamepad crashes on other values.

        Args:
            control (np.ndarray | None): ``[gas, brake, steer]``, or None to send nothing.
        """
        if control is not None:
            control = np.clip(control, -1.0, 1.0)
        super().send_control(control)

    def grab_lidar_speed_and_data(self):
        """Captures the LIDAR and keeps the OpenPlanet data so that ``info`` can be built from it.

        Returns:
            tuple[np.ndarray, np.ndarray, tuple]: LIDAR scan, speed and OpenPlanet data.
        """
        lidar, speed, self.data = super().grab_lidar_speed_and_data()
        return lidar, speed, self.data

    def reset(self, seed=None, options=None):
        """Restarts the race.

        Args:
            seed (int | None): Unused, the game is not seedable.
            options (dict | None): Unused.

        Returns:
            tuple[list, dict]: First observation and ``info``.
        """
        obs, _ = super().reset(seed, options)
        return obs, self.info()

    def get_obs_rew_terminated_info(self):
        """Captures the observation and computes the reward.

        Returns:
            tuple[list, np.float32, bool, dict]: Observation, reward, terminated flag and ``info``.
        """
        obs, rew, terminated, _ = super().get_obs_rew_terminated_info()
        return obs, rew, terminated, self.info()

    def info(self):
        """Builds the ``info`` dictionary from the last OpenPlanet data.

        Returns:
            dict: ``speed`` (m/s), ``end_of_track``, ``progress`` (0 to 1 along the trajectory),
                ``distance_m`` (along the trajectory) and ``dist_to_traj`` (m).
        """
        rf = self.reward_function
        pos = np.array(self.data[2:5])
        return {
            "speed": self.data[0],
            "end_of_track": bool(self.data[8]),
            "progress": rf.cur_idx / (rf.datalen - 1),
            "distance_m": rf.cur_idx * POINT_SPACING_M,
            "dist_to_traj": float(np.linalg.norm(pos - rf.data[rf.cur_idx])),
        }


class FlattenLidarObservation(gymnasium.ObservationWrapper):
    """Turns TMRL's ``(speed, lidar, action, action)`` tuple into the normalized vector of size 83.

    Args:
        env (gymnasium.Env): TMRL LIDAR environment.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=(83,), dtype=np.float32)

    def observation(self, observation):
        """Flattens and normalizes one observation.

        Args:
            observation (tuple): ``(speed, lidar, action, action)``.

        Returns:
            np.ndarray: Float32 vector of size 83.
        """
        speed, lidar, previous_action, last_action = observation
        return np.concatenate([
            speed / SPEED_SCALE,
            lidar.ravel() / LIDAR_SCALE,
            previous_action,
            last_action,
        ]).astype(np.float32)


def make_env(reward_path=DEFAULT_REWARD_PATH, ep_max_length=1000, flatten=True):
    """Creates the TrackMania environment.

    Args:
        reward_path (str | pathlib.Path): Demonstration trajectory (.pkl) of the track loaded in the game.
        ep_max_length (int): Number of steps before the episode is truncated.
        flatten (bool): If True, observations are vectors of size 83. Otherwise, TMRL's raw tuple is
            returned, which is needed to send None actions and drive manually.

    Returns:
        gymnasium.Env: The real-time environment.

    Raises:
        FileNotFoundError: If ``reward_path`` does not exist, because TMRL would otherwise silently use a
            dummy trajectory.
    """
    if not Path(reward_path).exists():
        raise FileNotFoundError(reward_path)
    config = dict(CONFIG_DICT)
    config["interface"] = LidarInterface
    config["interface_kwargs"] = {"reward_path": reward_path}
    config["ep_max_length"] = ep_max_length
    env = gymnasium.make(cfg.RTGYM_VERSION, config=config, disable_env_checker=True)
    return FlattenLidarObservation(env) if flatten else env
