"""Soft Actor-Critic with automatic entropy tuning (Haarnoja et al., 2018).

The agent works with any observation and action sizes, and with actions in [-1, 1]. It does not own the
replay buffer: the training loop samples batches and passes them to ``update``.
"""

import numpy as np
import torch


class SACAgent:
    """Soft Actor-Critic agent with two critics, their target copies and a learned temperature.

    It exposes ``act`` and ``reset`` like the baseline policies, so that it can be evaluated in the same way.

    Args:
        obs_dim (int): Size of an observation.
        act_dim (int): Size of an action.
        config (dict): Hyperparameters ``hidden``, ``lr``, ``gamma``, ``tau`` and ``device``. The target
            entropy is ``-act_dim``.
    """

    def __init__(self, obs_dim, act_dim, config):
        ...

    def act(self, obs, deterministic=False):
        """Chooses an action for one observation, without computing gradients.

        Args:
            obs (np.ndarray): Observation of shape ``(obs_dim,)``.
            deterministic (bool): If True, returns the mean action instead of a sample, for evaluation.

        Returns:
            np.ndarray: Float32 action of shape ``(act_dim,)``, in [-1, 1].
        """
        ...

    def reset(self):
        """Does nothing, because the agent has no state within an episode; it only exists for the common
        interface with the baseline policies."""

    def update(self, batch):
        """Performs one gradient step on the critics, the actor and the temperature, then updates the targets.

        Args:
            batch (dict): Float32 tensors ``obs`` ``(B, obs_dim)``, ``act`` ``(B, act_dim)``, ``rew`` ``(B,)``,
                ``next_obs`` ``(B, obs_dim)`` and ``terminated`` ``(B,)``, equal to 1.0 when the episode
                terminated and 0.0 otherwise.

        Returns:
            dict: ``critic_loss``, ``actor_loss``, ``alpha_loss``, ``alpha``, ``q_mean`` and ``entropy``, as
                floats.
        """
        ...

    def save(self, path):
        """Saves the networks, the targets, the optimizers and the temperature to a single file.

        Args:
            path (str | pathlib.Path): Destination ``.pt`` file.
        """
        ...

    def load(self, path):
        """Restores an agent saved with ``save``.

        Args:
            path (str | pathlib.Path): ``.pt`` file written by ``save``.
        """
        ...
