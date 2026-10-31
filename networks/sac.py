"""Neural networks of Soft Actor-Critic: a tanh-squashed Gaussian actor and a Q(s, a) critic."""

import torch
import torch.nn as nn


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
