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
    GAMEPAD_PLUG_TIMEOUT (float): Seconds to wait for Windows to install the virtual gamepad.
    GAMEPAD_POLL_DELAY (float): Seconds between two checks that the virtual gamepad is installed.
    VIGEM_ERROR_NONE (int): ViGEmBus code of a gamepad created without error.
    VIGEM_ERROR_TARGET_NOT_PLUGGED_IN (int): ViGEmBus code of a gamepad that Windows has not installed in time.
    VIGEM_TARGET_STATE_OFFSET (int): Offset in bytes of the state field in ViGEm's ``VIGEM_TARGET_T`` structure,
        after its ``ULONG Size`` and ``USHORT SerialNo`` fields.
    VIGEM_TARGET_CONNECTED (int): Value of that state field for a connected gamepad.
"""

import time
from pathlib import Path
import sys

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
GAMEPAD_PLUG_TIMEOUT = 30.0
GAMEPAD_POLL_DELAY = 0.2

VIGEM_ERROR_NONE = 0x20000000
VIGEM_ERROR_TARGET_NOT_PLUGGED_IN = 0xE0000007
VIGEM_TARGET_STATE_OFFSET = 8
VIGEM_TARGET_CONNECTED = 2


def gamepad_present(serial):
    """Tells whether Windows has finished installing the virtual Xbox 360 pad with this ViGEm serial.

    Args:
        serial (int): Serial number given by ViGEmBus, which is also the end of the USB instance ID.

    Returns:
        bool: True if the device is present.
    """
    if sys.platform != "win32":
        return True

    import ctypes

    devinst = ctypes.c_uint32()
    instance_id = f"USB\\VID_045E&PID_028E\\{serial:02d}"
    return ctypes.windll.cfgmgr32.CM_Locate_DevNodeW(ctypes.byref(devinst), instance_id, 0) == 0


def use_patient_gamepad():
    """Makes TMRL create a single virtual gamepad, waiting as long as Windows needs to install it.

    ViGEmBus gives up after 1 s when Windows is slow to install the virtual controller, which often happens
    on the first pad after a while without one. vgamepad then raises "The virtual device could not connect to
    ViGEmBus", but the pad does get plugged in a few seconds later. The client marked it as not connected,
    so it would refuse to send it any input or to remove it. Creating a new pad instead leaves the first one
    plugged: the game then listens to a pad that never moves, and the car stands still. So the pad is created
    once, Windows is polled until it is present, and the client is told it is connected. TMRL looks up
    ``vgamepad.VX360Gamepad`` when it initializes, so replacing it there is enough.

    Raises:
        RuntimeError: When the pad is created, if it is still absent after ``GAMEPAD_PLUG_TIMEOUT`` seconds.
    """
    if sys.platform != "win32":
        return

    import ctypes

    import vgamepad
    import vgamepad.win.vigem_client as vcli
    from vgamepad.win.virtual_gamepad import VBUS

    if getattr(vgamepad.VX360Gamepad, "patient", False):
        return

    class PatientX360Gamepad(vgamepad.VX360Gamepad):
        """Virtual Xbox 360 gamepad that waits for Windows to install it instead of failing after 1 s."""

        patient = True

        def __init__(self):
            """Creates the gamepad like ``VGamepad.__init__``, except when Windows is slow to install it.

            Raises:
                RuntimeError: If ViGEmBus fails for another reason, or if the gamepad is still absent after
                    ``GAMEPAD_PLUG_TIMEOUT`` seconds.
            """
            self.vbus = VBUS
            self._busp = VBUS.get_busp()
            self._devicep = self.target_alloc()
            self.CMPFUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ubyte,
                                            ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_void_p)
            self.cmp_func = None
            code = vcli.vigem_target_add(self._busp, self._devicep)
            if code == VIGEM_ERROR_TARGET_NOT_PLUGGED_IN:
                self.wait_until_plugged()
            elif code != VIGEM_ERROR_NONE:
                raise RuntimeError(f"ViGEmBus n'a pas pu créer la manette virtuelle (code {code:#x})")
            self.report = self.get_default_report()
            self.update()

        def wait_until_plugged(self):
            """Polls Windows until the gamepad is present, then marks it as connected in the ViGEm client.

            Raises:
                RuntimeError: If the gamepad is still absent after ``GAMEPAD_PLUG_TIMEOUT`` seconds.
            """
            serial = vcli.vigem_target_get_index(self._devicep)
            print(f"Manette virtuelle lente à s'installer, attente (n° {serial})", flush=True)
            deadline = time.time() + GAMEPAD_PLUG_TIMEOUT
            while not gamepad_present(serial):
                if time.time() > deadline:
                    raise RuntimeError(f"La manette virtuelle n° {serial} n'est pas apparue dans Windows "
                                       f"après {GAMEPAD_PLUG_TIMEOUT:.0f} s")
                time.sleep(GAMEPAD_POLL_DELAY)
            state = ctypes.c_int.from_address(self._devicep + VIGEM_TARGET_STATE_OFFSET)
            state.value = VIGEM_TARGET_CONNECTED
            print("Manette virtuelle connectée", flush=True)

    vgamepad.VX360Gamepad = PatientX360Gamepad

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
        use_patient_gamepad()
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


def make_env(reward_path=DEFAULT_REWARD_PATH, ep_max_length=3000, flatten=True):
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
