"""Baseline policies used as references for the trained agents.

Every policy exposes the same two methods, so that the evaluation script can run any of them, and later
the PPO and SAC agents, in the same way:

* ``act(obs)`` takes the observation of size 83 described in ``envs.trackmania`` and returns an action
  ``[gas, brake, steer]`` in [-1, 1];
* ``reset()`` is called at the start of each episode.
"""

import numpy as np


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
