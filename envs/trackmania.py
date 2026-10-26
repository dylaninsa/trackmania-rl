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
    DEFAULT_REWARD_PATH (pathlib.Path): Demonstration trajectory of the tmrl-test track.
"""

from pathlib import Path

import gymnasium
import numpy as np
import tmrl.config.config_constants as cfg
from tmrl.config.config_objects import CONFIG_DICT
from tmrl.custom.tm.tm_gym_interfaces import TM2020InterfaceLidar

from envs.reward import TrajectoryReward, load_trajectory

SPEED = slice(0, 1)
LIDAR = slice(1, 77)
LAST_LIDAR = slice(58, 77)
ACTIONS = slice(77, 83)

SPEED_SCALE = 100.0
LIDAR_SCALE = 500.0
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
        """Connects to the game, then replaces TMRL's reward function with one that reads our trajectory."""
        super().initialize()
        self.reward_function = TrajectoryReward(load_trajectory(self.reward_path))

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
        """Grabs the LIDAR and speed, computes the reward and builds the ``info`` dictionary.
        
        It does not call TMRL's version so that the finish signal can be passed to our reward.

        Returns:
            tuple[list, np.float32, bool, dict]: Observation, reward, terminated flag and ``info``.
        """
        lidar, speed, data = self.grab_lidar_speed_and_data()
        self.img_hist.append(lidar)
        reward, terminated = self.reward_function.compute_reward(data[2:5], finished=bool(data[8]))
        obs = [speed, np.array(list(self.img_hist), dtype=np.float32)]
        return obs, np.float32(reward), terminated, self.info()

    def info(self):
        """Builds the ``info`` dictionary from the last OpenPlanet data.

        Returns:
            dict: ``speed`` (m/s), ``end_of_track``, ``progress`` (0 to 1 along the trajectory),
                ``distance_m`` (along the trajectory) and ``dist_to_traj`` (m).
        """
        rf = self.reward_function
        return {
            "speed": self.data[0],
            "end_of_track": bool(self.data[8]),
            "progress": rf.progress,
            "distance_m": rf.distance_m,
            "dist_to_traj": rf.dist_to_traj,
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
