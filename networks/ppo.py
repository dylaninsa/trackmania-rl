"""Neural networks of PPO: a Gaussian actor with a state-independent standard deviation and a V(s) critic.

Both networks use two hidden layers with Tanh activations and an orthogonal initialization, the usual
choices for PPO.

Attributes:
    LOG_STD_MIN (float): Minimum log standard deviation of the Gaussian distribution.
    LOG_STD_MAX (float): Maximum log standard deviation of the Gaussian distribution.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def orthogonal_init(layer, gain=np.sqrt(2)):
    """Initializes a linear layer with orthogonal weights and zero biases.

    The usual gains are ``sqrt(2)`` for the hidden layers, 0.01 for the actor output, so that the first
    actions are close to the center, and 1.0 for the critic output.

    Args:
        layer (torch.nn.Module): Layer to initialize. Only ``nn.Linear`` layers are modified.
        gain (float): Scaling factor of the orthogonal weights.

    Returns:
        torch.nn.Module: The same layer, initialized.
    """
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)
    return layer


class ContinuousActor(nn.Module):
    """Stochastic policy: a diagonal Gaussian whose mean depends on the state.

    The mean is ``max_action * tanh(...)``, so it stays within the action bounds. The log standard deviation
    is a learned parameter that does not depend on the state, clipped to [``LOG_STD_MIN``, ``LOG_STD_MAX``].
    The sampled actions themselves are not squashed: the caller clips them before sending them to the
    environment.

    Args:
        state_dim (int): Size of an observation.
        action_dim (int): Size of an action.
        max_action (float): Bound of the mean, the largest absolute value of an action.
        hidden_dim (int): Number of units of each of the two hidden layers.
        init_log_std (float): Initial log standard deviation.
    """

    def __init__(self, state_dim, action_dim, max_action=2.0, hidden_dim=64, init_log_std=-0.5):
        super().__init__()
        self.action_dim = action_dim
        self.max_action = float(max_action)
        self.trunk = nn.Sequential(
            orthogonal_init(nn.Linear(state_dim, hidden_dim), gain=np.sqrt(2)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(hidden_dim, hidden_dim), gain=np.sqrt(2)),
            nn.Tanh(),
        )
        self.mu_head = orthogonal_init(nn.Linear(hidden_dim, action_dim), gain=0.01)
        self.log_std = nn.Parameter(torch.full((1, action_dim), init_log_std))

    def forward(self, state):
        """Computes the mean and the standard deviation of the policy.

        Args:
            state (torch.Tensor): Float32 observations of shape ``(B, state_dim)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Mean and standard deviation, both of shape ``(B, action_dim)``.
        """
        features = self.trunk(state)
        mu = self.max_action * torch.tanh(self.mu_head(features))
        log_std = torch.clamp(self.log_std, min=LOG_STD_MIN, max=LOG_STD_MAX).expand_as(mu)
        std = torch.exp(log_std)
        return mu, std

    def get_distribution(self, state):
        """Builds the Gaussian distribution of the policy.

        Args:
            state (torch.Tensor): Float32 observations of shape ``(B, state_dim)``.

        Returns:
            torch.distributions.Normal: The distribution ``N(mu(s), std^2)``.
        """
        mu, std = self.forward(state)
        return Normal(mu, std)

    def get_action_and_log_prob(self, state, deterministic=False):
        """Draws actions and computes their log-probability.

        Args:
            state (torch.Tensor): Float32 observations of shape ``(B, state_dim)``.
            deterministic (bool): If True, returns the mean instead of a sample, for evaluation.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Actions of shape ``(B, action_dim)``, and their
                log-probabilities summed over the action dimensions, of shape ``(B, 1)``.
        """
        dist = self.get_distribution(state)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob

    def evaluate_actions(self, states, actions):
        """Computes the log-probability and the entropy of actions already collected.

        Used during the PPO update, to compare the current policy with the one that collected the data.

        Args:
            states (torch.Tensor): Float32 observations of shape ``(B, state_dim)``.
            actions (torch.Tensor): Float32 actions of shape ``(B, action_dim)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Log-probabilities and entropies, both of shape ``(B, 1)``.
        """
        dist = self.get_distribution(states)
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_probs, entropy


class Critic(nn.Module):
    """State-value function V(s): the expected discounted return from state ``s`` when following the policy.

    Args:
        state_dim (int): Size of an observation.
        hidden_dim (int): Number of units of each of the two hidden layers.
    """

    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            orthogonal_init(nn.Linear(state_dim, hidden_dim), gain=np.sqrt(2)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(hidden_dim, hidden_dim), gain=np.sqrt(2)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(hidden_dim, 1), gain=1.0),
        )

    def forward(self, state):
        """Estimates V for a batch of observations.

        Args:
            state (torch.Tensor): Float32 observations of shape ``(B, state_dim)``.

        Returns:
            torch.Tensor: Values of shape ``(B, 1)``.
        """
        return self.net(state)
