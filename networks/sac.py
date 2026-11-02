"""Neural networks of Soft Actor-Critic: a tanh-squashed Gaussian actor and a Q(s, a) critic.

Attributes:
    LOG_STD_MIN (float): Minimum log standard deviation of the Gaussian distribution.
    LOG_STD_MAX (float): Maximum log standard deviation of the Gaussian distribution.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -20
LOG_STD_MAX = 2


def mlp(in_dim, out_dim, hidden):
    """Creates a multi-layer perceptron with two hidden layers and ReLU activations.

    Args:
        in_dim (int): Size of the input.
        out_dim (int): Size of the output.
        hidden (int): Number of units of each of the two hidden layers.

    Returns:
        torch.nn.Sequential: The MLP.
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class Actor(nn.Module):
    """Stochastic policy: a Gaussian whose sample is squashed by tanh into [-1, 1].

    A single MLP outputs, side by side, the mean and the log standard deviation of each action dimension.
    The log standard deviation is clipped to [``LOG_STD_MIN``, ``LOG_STD_MAX``]: a vanishing standard deviation
    would make the log-probability explode, and a huge one would push every action to ±1.

    Args:
        obs_dim (int): Size of an observation.
        act_dim (int): Size of an action.
        hidden (int): Number of units of each of the two hidden layers.
    """

    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim, hidden)

    def forward(self, obs, deterministic=False):
        """Draws actions and computes their log-probability.

        The sample ``u`` uses the reparameterization trick (``rsample``), so that gradients flow from the
        action back to the actor's weights. The log-probability is corrected for the tanh squashing with
        ``log(1 - tanh(u)^2) = 2 * (log(2) - u - softplus(-2u))``, which stays finite even when ``tanh(u)``
        rounds to ±1 in float32.

        Args:
            obs (torch.Tensor): Float32 observations of shape ``(B, obs_dim)``.
            deterministic (bool): If True, uses the mean instead of a sample, for evaluation.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Actions of shape ``(B, act_dim)`` in [-1, 1], and their
                log-probabilities of shape ``(B,)``.
        """
        mu, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        if deterministic:
            u = mu
        else:
            u = dist.rsample()
        log_prob = dist.log_prob(u).sum(dim=-1)
        log_prob_corrected = log_prob - 2 * (math.log(2) - u - F.softplus(-2 * u)).sum(dim=-1)
        return torch.tanh(u), log_prob_corrected


class Critic(nn.Module):
    """Action-value function Q(s, a): the expected discounted return, entropy bonus included, of taking
    action ``a`` in state ``s`` and then following the policy.

    The observation and the action are concatenated and fed to a single MLP with one output. SAC uses two
    such critics and their two target copies, all created by the agent.

    Args:
        obs_dim (int): Size of an observation.
        act_dim (int): Size of an action.
        hidden (int): Number of units of each of the two hidden layers.
    """

    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, obs, act):
        """Estimates Q for a batch of observation-action pairs.

        The last dimension of the MLP output is squeezed so that Q has the same shape as the rewards and the
        Bellman target. A ``(B, 1)`` output would silently broadcast against a ``(B,)`` target into a
        ``(B, B)`` matrix and make the loss wrong without raising any error.

        Args:
            obs (torch.Tensor): Float32 observations of shape ``(B, obs_dim)``.
            act (torch.Tensor): Float32 actions of shape ``(B, act_dim)``.

        Returns:
            torch.Tensor: Q values of shape ``(B,)``.
        """
        x = torch.cat([obs, act], dim=-1)
        return self.net(x).squeeze(-1)
