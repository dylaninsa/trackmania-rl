"""Replay buffer storing the transitions collected by the agent, for off-policy algorithms such as SAC."""

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size circular buffer: once full, new transitions overwrite the oldest ones.

    Only ``terminated`` is stored, not ``truncated``: a truncated episode could have continued, so its last
    transition must still be bootstrapped.

    Args:
        obs_dim (int): Size of an observation.
        act_dim (int): Size of an action.
        capacity (int): Maximum number of transitions.
        device (str): Device of the tensors returned by ``sample``.
    """

    def __init__(self, obs_dim, act_dim, capacity, device="cpu"):
        ...

    def add(self, obs, act, rew, next_obs, terminated):
        """Stores one transition.

        Args:
            obs (np.ndarray): Observation before the action.
            act (np.ndarray): Action taken.
            rew (float): Reward received.
            next_obs (np.ndarray): Observation after the action.
            terminated (bool): Whether the episode terminated after this transition.
        """
        ...

    def sample(self, batch_size):
        """Draws transitions uniformly at random.

        Args:
            batch_size (int): Number of transitions.

        Returns:
            dict: Float32 tensors ``obs``, ``act``, ``rew``, ``next_obs`` and ``terminated``, in the format
                expected by ``SACAgent.update``.
        """
        ...

    def __len__(self):
        """Returns the number of transitions currently stored."""
        ...
