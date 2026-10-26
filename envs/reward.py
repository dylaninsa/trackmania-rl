"""Reward based on the progress of the car along a demonstration trajectory.

The demonstration trajectory is a lap recorded with ``python -m tmrl --record-reward``, stored as points
spaced ``POINT_SPACING_M`` apart. At each step, the car is matched to the closest point ahead of its current
point, and the reward is the number of points passed since the previous step, divided by 100.

Attributes:
    POINT_SPACING_M (float): Distance in meters between two consecutive points of a trajectory.
"""

import pickle

import numpy as np

POINT_SPACING_M = 0.1


def load_trajectory(path):
    """Loads a demonstration trajectory.

    Args:
        path (str | pathlib.Path): Trajectory file (.pkl) recorded with ``python -m tmrl --record-reward``.

    Returns:
        np.ndarray: Array of shape ``(N, 3)``, one ``(x, y, z)`` point per row.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


class TrajectoryReward:
    """Rewards the progress of the car along a demonstration trajectory.

    Args:
        trajectory (np.ndarray): Points of the trajectory, of shape ``(N, 3)``.
        max_forward_points (int): Number of points ahead of the current point in which the car is searched.
        max_distance_m (float): The episode terminates when the car is further than this from the trajectory.
        patience (int): The episode terminates after this many consecutive steps without progress.
        grace_steps (int): Number of steps at the start of an episode during which a lack of progress is not
            counted, because the car starts from a standstill.
        finish_bonus (float): Bonus reward for completing the trajectory.
    """

    def __init__(self, trajectory, max_forward_points=500, max_distance_m=30.0, patience=10, grace_steps=70,
                 finish_bonus=100.0):
        self.trajectory = trajectory
        self.max_forward_points = max_forward_points
        self.max_distance_m = max_distance_m
        self.patience = patience
        self.grace_steps = grace_steps
        self.finish_bonus = finish_bonus
        self.reset()

    def reset(self):
        """Resets the state of the episode: the car is back at the first point of the trajectory."""
        self.index = 0
        self.steps = 0
        self.steps_without_progress = 0
        self.dist_to_traj = 0.0

    def nearest_point(self, pos):
        """Finds the point of the trajectory closest to the car, among the points ahead of the current one.

        Only the points from ``self.index`` to ``self.index + max_forward_points`` are considered, so the car
        can never move backwards along the trajectory.

        Args:
            pos (np.ndarray): Position ``(x, y, z)`` of the car.

        Returns:
            tuple[int, float]: Index of the closest point and its distance to the car in meters.
        """
        start = self.index
        end = min(self.index + self.max_forward_points, len(self.trajectory))
        window = self.trajectory[start:end]
        distances = np.linalg.norm(window - pos, axis=1)
        nearest_idx = np.argmin(distances)
        return int(start + nearest_idx), float(distances[nearest_idx])

    def compute_reward(self, pos, finished=False):
        """Computes the reward of one step and whether the episode must terminate.

        Args:
            pos (np.ndarray): Position ``(x, y, z)`` of the car.
            finished (bool): Whether the car has reached the end of the trajectory.

        Returns:
            tuple[float, bool]: Reward (number of points passed divided by 100, and bonus for finishing)
                and terminated flag.
        """
        self.steps += 1
        nearest_idx, dist_to_traj = self.nearest_point(pos)
        self.dist_to_traj = dist_to_traj
        if dist_to_traj > self.max_distance_m:
            return 0.0, True
        reward = (nearest_idx - self.index) / 100.0
        if nearest_idx > self.index:
            self.steps_without_progress = 0
        elif self.steps > self.grace_steps:
            self.steps_without_progress += 1
        self.index = nearest_idx
        if finished:
            return reward + self.finish_bonus, True
        if self.steps_without_progress >= self.patience:
            return reward, True
        return reward, False

    @property
    def progress(self):
        """float: Fraction of the trajectory covered, from 0 at the start to 1 at the end."""
        return self.index / (len(self.trajectory) - 1)

    @property
    def distance_m(self):
        """float: Distance covered along the trajectory, in meters."""
        return self.index * POINT_SPACING_M
