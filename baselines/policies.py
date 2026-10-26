"""Baseline policies used as references for the trained agents.

Every policy exposes the same two methods, so that the evaluation script can run any of them, and later
the PPO and SAC agents, in the same way:

* ``act(obs)`` takes the observation of size 83 described in ``envs.trackmania`` and returns an action
  ``[gas, brake, steer]`` in [-1, 1];
* ``reset()`` is called at the start of each episode.
"""

import numpy as np

from envs.trackmania import LAST_LIDAR, SPEED, SPEED_SCALE


class RandomPolicy:
    """Policy that draws every action uniformly at random, without looking at the observation.

    It is the lowest reference of the comparison: an agent that does not beat it has learned nothing.

    Args:
        seed (int | None): Seed of the random generator. The same seed gives the same sequence of actions,
            which is what allows the random agent to be evaluated on several seeds. With None, the actions
            change at every run.
    """

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """Does nothing, because the random policy has no state; it only exists for the common interface."""

    def act(self, obs):
        """Draws a random action.

        Gas and brake only act when they are positive, so each is active half of the time and the car often
        accelerates and brakes at once. The random agent therefore barely moves, which is expected from a
        lower bound.

        Args:
            obs (np.ndarray): Observation of size 83, ignored.

        Returns:
            np.ndarray: Float32 action ``[gas, brake, steer]``, each value drawn uniformly in [-1, 1].
        """
        return self.rng.uniform(low=-1.0, high=1.0, size=(3,)).astype(np.float32)


class PIDPolicy:
    """Heuristic that follows the center of the track using the most recent LIDAR scan.

    The steering is a PID controller on the error "free space on the right minus free space on the left":
    the error is zero when the car is centered, and it becomes positive before a right turn, because the
    right beams see further than the left ones. Gas and brake follow a simple rule instead: the more free
    space straight ahead, the higher the target speed.

    Args:
        kp (float): Proportional gain of the steering: the further off-center, the stronger the steering.
        ki (float): Integral gain of the steering, which corrects a lasting offset, for example in a long turn.
        kd (float): Derivative gain of the steering, which damps the correction to avoid zigzagging.
        min_speed (float): Target speed in m/s when a wall is right in front of the car.
        max_speed (float): Target speed in m/s when the road ahead is clear.
        brake_margin (float): The car only brakes when it exceeds the target speed by more than this, in m/s,
            so that it does not alternate between full gas and full brake at every step.
        dt (float): Duration of a step in seconds, used by the integral and derivative terms.
    """

    def __init__(self, kp=3.0, ki=0.0, kd=0.1, min_speed=10.0, max_speed=60.0, brake_margin=5.0, dt=0.05):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.brake_margin = brake_margin
        self.dt = dt

        self.reset()

    def reset(self):
        """Resets the state of the PID controller at the start of an episode.

        The integral is cleared so that the error accumulated in one episode does not affect the next one.
        The previous error is set to None rather than 0, so that the first step uses a zero derivative
        instead of a sudden steering kick.
        """
        self.integral = 0.0
        self.previous_error = None

    def steer(self, lidar):
        """Computes the steering command with the PID controller and updates its state.

        The error is the mean of the right beams (0 to 8) minus the mean of the left beams (10 to 18). The
        front beam (9) belongs to neither side. The command is clipped to [-1, 1] so that the method returns
        a valid steering value on its own, whatever the gains.

        Args:
            lidar (np.ndarray): The 19 normalized distances of the most recent LIDAR scan.

        Returns:
            float: Steering command in [-1, 1], positive to the right.
        """
        error = np.mean(lidar[0:9]) - np.mean(lidar[10:19])
        self.integral += error * self.dt
        derivative = 0 if self.previous_error is None else (error - self.previous_error) / self.dt
        self.previous_error = error
        return float(np.clip(self.kp * error + self.ki * self.integral + self.kd * derivative, -1.0, 1.0))

    def throttle(self, lidar, speed):
        """Chooses gas and brake from the free space straight ahead.

        The target speed grows linearly from ``min_speed`` to ``max_speed`` with the front beam (9), clipped to
        [0, 1]. The car accelerates below the target speed, brakes above the target speed plus
        ``brake_margin``, and coasts in between.

        Args:
            lidar (np.ndarray): The 19 normalized distances of the most recent LIDAR scan.
            speed (float): Speed of the car in m/s.

        Returns:
            tuple[float, float]: Gas and brake, each 0.0 or 1.0.
        """
        front_distance = np.clip(lidar[9], 0.0, 1.0)
        target_speed = self.min_speed + (self.max_speed - self.min_speed) * front_distance
        if speed < target_speed:
            return 1.0, 0.0
        if speed > target_speed + self.brake_margin:
            return 0.0, 1.0
        return 0.0, 0.0

    def act(self, obs):
        """Computes the action from the most recent LIDAR scan and the speed.

        Args:
            obs (np.ndarray): Observation of size 83. The speed is converted back to m/s with ``SPEED_SCALE``.

        Returns:
            np.ndarray: Float32 action ``[gas, brake, steer]``.
        """
        last_lidar = obs[LAST_LIDAR]
        speed = float(obs[SPEED][0] * SPEED_SCALE)
        steer = self.steer(last_lidar)
        gas, brake = self.throttle(last_lidar, speed)
        return np.array([gas, brake, steer], dtype=np.float32)
