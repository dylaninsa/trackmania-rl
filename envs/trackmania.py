"""Création de l'environnement TrackMania 2020 (LIDAR) utilisé par tout le projet."""

from pathlib import Path

from envs.wrappers import ACT_BUF_LEN, IMG_HIST_LEN, LIDAR_SCALE, SPEED_SCALE, FlattenLidarObservation

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REWARD_PATH = REPO_ROOT / "data" / "reward" / "tmrl-test.pkl"


def make_env(reward_path=DEFAULT_REWARD_PATH, ep_max_length=1000, flatten=True,
             speed_scale=SPEED_SCALE, lidar_scale=LIDAR_SCALE):
    """Crée l'environnement Gymnasium TrackMania en mode LIDAR.

    Args:
        reward_path: trajectoire de démonstration (.pkl) de la piste chargée dans le jeu.
        ep_max_length: nombre de pas (0,05 s chacun) avant troncation de l'épisode.
        flatten: si True, l'observation est un vecteur float32 de dimension 83 (voir envs/wrappers.py) ;
            sinon, le tuple brut (vitesse, LIDAR, action, action) de TMRL.
        speed_scale, lidar_scale: constantes de normalisation (seulement si flatten=True).
    """
    reward_path = Path(reward_path)
    if not reward_path.exists():
        # TMRL utiliserait silencieusement une trajectoire factice : on préfère échouer.
        raise FileNotFoundError(f"Trajectoire de démonstration introuvable : {reward_path}")

    # Imports différés : TMRL plante à l'import si TmrlData n'existe pas (machines sans le jeu).
    import gymnasium
    import tmrl.config.config_constants as cfg
    from tmrl.config.config_objects import CONFIG_DICT

    from envs.tm_interface import LidarInterface

    problems = []
    if cfg.RTGYM_INTERFACE != "TM20LIDAR":
        problems.append(f'"RTGYM_INTERFACE" vaut "{cfg.RTGYM_INTERFACE}" au lieu de "TM20LIDAR"')
    if cfg.IMG_HIST_LEN != IMG_HIST_LEN:
        problems.append(f'"IMG_HIST_LEN" vaut {cfg.IMG_HIST_LEN} au lieu de {IMG_HIST_LEN}')
    if cfg.ACT_BUF_LEN != ACT_BUF_LEN:
        problems.append(f'"act_buf_len" vaut {cfg.ACT_BUF_LEN} au lieu de {ACT_BUF_LEN}')
    if not cfg.PRAGMA_GAMEPAD:
        problems.append('"VIRTUAL_GAMEPAD" est false (les actions deviendraient binaires)')
    if problems:
        raise ValueError(f"{cfg.CONFIG_FILE} n'est pas configuré pour ce projet :\n- " + "\n- ".join(problems))

    config = dict(CONFIG_DICT)
    config["interface"] = LidarInterface
    config["interface_kwargs"] = {"reward_path": reward_path, "img_hist_len": IMG_HIST_LEN, "gamepad": True}
    config["ep_max_length"] = ep_max_length  # rtgym vaut np.inf par défaut : jamais de troncation

    env = gymnasium.make(cfg.RTGYM_VERSION, config=config, disable_env_checker=True)
    if flatten:
        env = FlattenLidarObservation(env, speed_scale=speed_scale, lidar_scale=lidar_scale)
    return env
