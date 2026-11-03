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
        seed (int | None): Seed of the random generator used by ``sample``, so that runs can be reproduced.
    """

    def __init__(self, obs_dim, act_dim, capacity, device="cpu", seed=None):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(self.capacity, dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros(self.capacity, dtype=np.float32)
        self._ptr = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def add(self, obs, act, rew, next_obs, terminated):
        """Stores one transition.

        Args:
            obs (np.ndarray): Observation before the action.
            act (np.ndarray): Action taken.
            rew (float): Reward received.
            next_obs (np.ndarray): Observation after the action.
            terminated (bool): Whether the episode terminated after this transition.
        """
        i = self._ptr
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        self.terminated[i] = float(terminated)
        self._ptr = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size):
        """Draws transitions uniformly at random.

        Args:
            batch_size (int): Number of transitions.

        Returns:
            dict: Float32 tensors ``obs``, ``act``, ``rew``, ``next_obs`` and ``terminated``, in the format
                expected by ``SACAgent.update``.
        """
        if self._size == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        idx = self._rng.integers(0, self._size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx], device=self.device),
            "act": torch.as_tensor(self.act[idx], device=self.device),
            "rew": torch.as_tensor(self.rew[idx], device=self.device),  # shape (B,)
            "next_obs": torch.as_tensor(self.next_obs[idx], device=self.device),
            "terminated": torch.as_tensor(self.terminated[idx], device=self.device),  # shape (B,)
        }

    def __len__(self):
        """Returns the number of transitions currently stored."""
        return self._size