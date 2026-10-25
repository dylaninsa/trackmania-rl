"""Formatting of the LIDAR observation, independent of TMRL and testable without the game.

Attributes:
    IMG_HIST_LEN (int): Number of LIDAR scans in the observation.
    NB_BEAMS (int): Number of beams per scan. Beam 0 points right, beam 9 straight ahead, beam 18 left.
    ACT_BUF_LEN (int): Number of past actions in the observation.
    ACT_DIM (int): Size of an action, ``[gas, brake, steer]``.
    SPEED (slice): Speed in the flattened observation.
    LIDAR (slice): All LIDAR scans in the flattened observation, oldest first.
    LAST_LIDAR (slice): Most recent LIDAR scan in the flattened observation.
    ACTIONS (slice): Past actions in the flattened observation, oldest first.
    OBS_DIM (int): Size of the flattened observation (83).
    SPEED_SCALE (float): Speed normalization constant. The speed is in m/s (a manual lap of tmrl-test covered
        2197 m in 43.2 s while the reported speed averaged about 50) and peaks around 75 on that track.
    LIDAR_SCALE (float): LIDAR normalization constant. Geometric bound: a beam is at most about 460 px long
        in a 958x488 window.
"""

import gymnasium
import numpy as np

IMG_HIST_LEN = 4
NB_BEAMS = 19
ACT_BUF_LEN = 2
ACT_DIM = 3

SPEED = slice(0, 1)
LIDAR = slice(1, 1 + IMG_HIST_LEN * NB_BEAMS)
LAST_LIDAR = slice(LIDAR.stop - NB_BEAMS, LIDAR.stop)
ACTIONS = slice(LIDAR.stop, LIDAR.stop + ACT_BUF_LEN * ACT_DIM)
OBS_DIM = ACTIONS.stop

SPEED_SCALE = 100.0
LIDAR_SCALE = 500.0


class FlattenLidarObservation(gymnasium.ObservationWrapper):
    """Turns TMRL's ``(speed, lidar, action, action)`` tuple into a float32 vector of size ``OBS_DIM``.

    Args:
        env (gymnasium.Env): Environment whose observation space is TMRL's LIDAR tuple.
        speed_scale (float): The speed is divided by this value.
        lidar_scale (float): LIDAR distances are divided by this value.

    Raises:
        ValueError: If the observation space does not have ``IMG_HIST_LEN`` scans of ``NB_BEAMS`` beams
            and ``ACT_BUF_LEN`` actions.
    """

    def __init__(self, env, speed_scale=SPEED_SCALE, lidar_scale=LIDAR_SCALE):
        super().__init__(env)
        speed_space, lidar_space, *act_spaces = env.observation_space.spaces
        if lidar_space.shape != (IMG_HIST_LEN, NB_BEAMS) or len(act_spaces) != ACT_BUF_LEN:
            raise ValueError(
                f"Observation inattendue : LIDAR {lidar_space.shape}, {len(act_spaces)} actions. "
                f"Attendu : LIDAR ({IMG_HIST_LEN}, {NB_BEAMS}) et {ACT_BUF_LEN} actions."
            )
        self.speed_scale = speed_scale
        self.lidar_scale = lidar_scale
        low = np.zeros(OBS_DIM, dtype=np.float32)
        high = np.full(OBS_DIM, np.inf, dtype=np.float32)
        low[ACTIONS], high[ACTIONS] = -1.0, 1.0
        self.observation_space = gymnasium.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):
        """Flattens and normalizes one observation.

        Args:
            observation (tuple): TMRL observation ``(speed, lidar, action, action)``.

        Returns:
            np.ndarray: Float32 vector of size ``OBS_DIM``, laid out as described by the module slices.
        """
        speed, lidar, *actions = observation
        return np.concatenate([
            np.asarray(speed, dtype=np.float32).ravel() / self.speed_scale,
            np.asarray(lidar, dtype=np.float32).ravel() / self.lidar_scale,
            *(np.asarray(a, dtype=np.float32).ravel() for a in actions),
        ])
