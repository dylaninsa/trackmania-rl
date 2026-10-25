"""Mise en forme de l'observation LIDAR (indépendant de TMRL, testable sans le jeu)."""

import gymnasium
import numpy as np

IMG_HIST_LEN = 4  # nombre de balayages LIDAR dans l'observation
NB_BEAMS = 19  # faisceaux par balayage (index 0 = droite, 9 = devant, 18 = gauche)
ACT_BUF_LEN = 2  # nombre d'actions passées dans l'observation
ACT_DIM = 3  # [gaz, frein, direction]

# Disposition du vecteur d'observation aplati (dimension OBS_DIM = 83)
SPEED = slice(0, 1)
LIDAR = slice(1, 1 + IMG_HIST_LEN * NB_BEAMS)  # balayage le plus ancien en premier
LAST_LIDAR = slice(LIDAR.stop - NB_BEAMS, LIDAR.stop)  # balayage le plus récent
ACTIONS = slice(LIDAR.stop, LIDAR.stop + ACT_BUF_LEN * ACT_DIM)  # action la plus ancienne en premier
OBS_DIM = ACTIONS.stop

# Constantes de normalisation, à calibrer avec scripts/check_env.py
SPEED_SCALE = 1000.0  # borne haute de la vitesse dans l'observation_space de TMRL
LIDAR_SCALE = 500.0  # borne géométrique : un faisceau mesure au plus ~460 px dans une fenêtre 958x488


class FlattenLidarObservation(gymnasium.ObservationWrapper):
    """Transforme le tuple (vitesse, LIDAR, action, action) de TMRL en un vecteur float32 de dimension 83."""

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
        speed, lidar, *actions = observation
        return np.concatenate([
            np.asarray(speed, dtype=np.float32).ravel() / self.speed_scale,
            np.asarray(lidar, dtype=np.float32).ravel() / self.lidar_scale,
            *(np.asarray(a, dtype=np.float32).ravel() for a in actions),
        ])
