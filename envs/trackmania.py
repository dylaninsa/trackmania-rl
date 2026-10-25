"""Creation of the TrackMania 2020 LIDAR environment used by the whole project.

Attributes:
    REPO_ROOT (pathlib.Path): Root of the repository.
    DEFAULT_REWARD_PATH (pathlib.Path): Demonstration trajectory of the ``tmrl-test`` track.
"""

from pathlib import Path

from envs.wrappers import ACT_BUF_LEN, IMG_HIST_LEN, LIDAR_SCALE, SPEED_SCALE, FlattenLidarObservation

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REWARD_PATH = REPO_ROOT / "data" / "reward" / "tmrl-test.pkl"


def make_env(reward_path=DEFAULT_REWARD_PATH, ep_max_length=1000, flatten=True,
             speed_scale=SPEED_SCALE, lidar_scale=LIDAR_SCALE):
    """Creates the TrackMania Gymnasium environment in LIDAR mode.

    TMRL is imported inside this function because importing it fails on machines without the TmrlData
    folder, which would break code that only needs ``envs.wrappers``.

    Args:
        reward_path (str | pathlib.Path): Demonstration trajectory (.pkl) of the track loaded in the game.
        ep_max_length (int): Number of steps (0.05 s each) before the episode is truncated. It must be set
            explicitly because rtgym defaults to ``np.inf``, which never truncates.
        flatten (bool): If True, observations are float32 vectors of size 83 (see ``envs.wrappers``).
            Otherwise, TMRL's raw ``(speed, lidar, action, action)`` tuple is returned.
        speed_scale (float): Speed normalization constant, used only if ``flatten`` is True.
        lidar_scale (float): LIDAR normalization constant, used only if ``flatten`` is True.

    Returns:
        gymnasium.Env: The real-time environment.

    Raises:
        FileNotFoundError: If ``reward_path`` does not exist. TMRL would otherwise silently use a dummy
            trajectory.
        ValueError: If ``TmrlData/config/config.json`` is not configured for this project.
    """
    reward_path = Path(reward_path)
    if not reward_path.exists():
        raise FileNotFoundError(f"Trajectoire de démonstration introuvable : {reward_path}")

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
    config["ep_max_length"] = ep_max_length

    env = gymnasium.make(cfg.RTGYM_VERSION, config=config, disable_env_checker=True)
    if flatten:
        env = FlattenLidarObservation(env, speed_scale=speed_scale, lidar_scale=lidar_scale)
    return env
