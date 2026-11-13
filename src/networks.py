"""
Module 2 : Les réseaux Acteur et Critique en PyTorch (Contrôle Continu).

Dans un problème à espace d'actions continu :
- Pendulum-v1 : action de dimension 1 dans [-2.0, +2.0] (couple appliqué au pendule)
- TMRL (TrackMania) : action de dimension 3 dans [-1.0, +1.0]^3 (gaz, frein, direction)

L'Acteur ne prédit pas des probabilités discrètes (Softmax), mais les paramètres
d'une distribution Gaussienne multidimensionnelle :
    pi_theta(a | s) = N( mu_theta(s), diag(sigma_theta^2) )

Le Critique estime la fonction de valeur d'état V_phi(s) (espérance des récompenses
futures actualisées depuis l'état s), indispensable pour calculer l'avantage GAE.
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def orthogonal_init(layer: nn.Module, gain: float = np.sqrt(2)) -> nn.Module:
    """
    Initialisation orthogonale des poids (bonne pratique standard pour PPO).
    - Couches cachées : gain = sqrt(2) (adapté à l'activation Tanh)
    - Sortie de l'Acteur : gain = 0.01 (pour que les premières actions soient proches de 0)
    - Sortie du Critique : gain = 1.0
    """
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)
    return layer


class ContinuousActor(nn.Module):
    """
    Réseau Acteur (Politique stochastique continue) : pi_theta(a | s).

    Architecture :
        État s (dim: state_dim)
          -> Linear(state_dim, hidden_dim) + Tanh
          -> Linear(hidden_dim, hidden_dim) + Tanh
          -> mu(s) = max_action * Tanh(Linear(hidden_dim, action_dim))
        + paramètre appris log_std (dim: action_dim)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 2.0,
        hidden_dim: int = 64,
        init_log_std: float = -0.5,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.max_action = float(max_action)

        self. trunk = nn.Sequential(
            orthogonal_init(nn.Linear(state_dim, hidden_dim), gain=np.sqrt(2)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(hidden_dim, hidden_dim), gain=np.sqrt(2)),
            nn.Tanh(),
        )
        self.mu_head = orthogonal_init(nn.Linear(hidden_dim, action_dim), gain=0.01)

        # Log de l'écart-type (indépendant de l'état, très stable en PPO continu)
        self.log_std = nn.Parameter(torch.full((1, action_dim), init_log_std))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcule la moyenne mu(s) et l'écart-type std de la politique Gaussienne.
        """
        features = self.trunk(state)
        mu = self.max_action * torch.tanh(self.mu_head(features))
        # On borne log_std pour éviter un écart-type trop faible ou explosif
        log_std = torch.clamp(self.log_std, min=-5.0, max=2.0).expand_as(mu)
        std = torch.exp(log_std)
        return mu, std

    def get_distribution(self, state: torch.Tensor) -> Normal:
        """Retourne la distribution Gaussienne N(mu(s), std^2)."""
        mu, std = self.forward(state)
        return Normal(mu, std)

    def get_action_and_log_prob(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Échantillonne une action à partir de l'état s et renvoie (action, log_prob).
        En mode déterministe (évaluation), renvoie directement la moyenne mu(s).
        """
        dist = self.get_distribution(state)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()

        # Somme des log-probabilités sur toutes les dimensions de l'action
        # (1D sur Pendulum, 3D sur TrackMania : gaz + frein + direction)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Utilisé pendant la mise à jour PPO sur les mini-batchs :
        Calcule log pi_theta(a | s) et l'entropie H(pi_theta(. | s)) pour des couples
        (states, actions) déjà collectés dans le RolloutBuffer.
        """
        dist = self.get_distribution(states)
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_probs, entropy


class Critic(nn.Module):
    """
    Réseau Critique (Fonction de valeur d'état) : V_phi(s).

    Prédit un scalaire estimant la somme espérée des récompenses actualisées
    à partir de l'état s : V(s_t) = E[ sum_{k=0..inf} gamma^k * r_{t+k} ].
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            orthogonal_init(nn.Linear(state_dim, hidden_dim), gain=np.sqrt(2)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(hidden_dim, hidden_dim), gain=np.sqrt(2)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(hidden_dim, 1), gain=1.0),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
