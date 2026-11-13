"""
Module 3 : L'algorithme PPO (Proximal Policy Optimization).

Référence : Schulman et al. (2017) - "Proximal Policy Optimization Algorithms"

Ce module contient :
1. RolloutBuffer : Stocke une trajectoire de N pas de temps (on-policy) et calcule
   les avantages GAE (Generalized Advantage Estimation) ainsi que les retours cibles.
2. PPOAgent : Regroupe l'Acteur, le Critique, la normalisation d'état et la mise à
   jour par mini-batchs avec l'objectif clippé L^{CLIP}.
"""

from typing import Any, Dict, Generator, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks.ppo import ContinuousActor, Critic


class RunningMeanStd:
    """
    Calcule de manière incrémentale (algorithme de Welford) la moyenne et la variance
    des observations pour les normaliser : s_norm = (s - mean) / sqrt(var + eps).
    
    C'est essentiel en PPO continu :
    - Sur Pendulum-v1 : vitesse angulaire dans [-8, 8] vs cos/sin dans [-1, 1].
    - Sur TMRL : vitesse dans [0, 100+] vs LIDAR vs actions dans [-1, 1].
    """

    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: np.ndarray, clip: float = 10.0, center: bool = False) -> np.ndarray:
        """
        Met à l'échelle l'observation par l'écart-type estimé.
        Par défaut `center=False` pour préserver le zéro physique des capteurs
        symétriques (ex: sin(theta)=0 à la verticale sur Pendulum, direction=0 sur TMRL).
        """
        if center:
            norm_x = (x - self.mean) / np.sqrt(self.var + 1e-8)
        else:
            norm_x = x / np.sqrt(self.var + 1e-8)
        return np.clip(norm_x, -clip, clip).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.mean = state_dict["mean"]
        self.var = state_dict["var"]
        self.count = state_dict["count"]


class RolloutBuffer:
    """
    Mémoire tampon On-Policy pour PPO.
    Contrairement à SAC (Off-Policy, qui garde un immense Replay Buffer), PPO collecte
    un horizon fixe de `buffer_size` transitions avec la politique courante, effectue
    K époques d'optimisation, puis vide le buffer (`clear`) avant la collecte suivante.
    """

    def __init__(self, buffer_size: int, state_dim: int, action_dim: int, device: torch.device):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.clear()

    def clear(self) -> None:
        self.states = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.next_values = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.terminateds = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, 1), dtype=np.float32)

        self.advantages = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.ptr = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        next_value: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Ajoute une transition (s_t, a_t, log_pi, r_t, V(s_t), V(s_{t+1}), terminated, done)."""
        if self.ptr >= self.buffer_size:
            raise RuntimeError("RolloutBuffer est plein ! Appelez update() puis clear().")

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.next_values[self.ptr] = next_value
        self.terminateds[self.ptr] = float(terminated)
        self.dones[self.ptr] = float(terminated or truncated)
        self.ptr += 1

    def compute_returns_and_advantages(self, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
        """
        Calcule l'estimation d'avantage généralisé (GAE - Schulman et al., 2015) :
            delta_t = r_t + gamma * (1 - terminated_t) * V(s_{t+1}) - V(s_t)
            A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
            R_t = A_t + V(s_t)

        Note importante (Pendulum-v1 & TMRL) :
        - Si un épisode se termine par limite de temps (`truncated=True`, `terminated=False`),
          l'état final n'est pas un crash : on bootstrappe donc avec V(s_{t+1}).
        - En revanche, comme l'environnement est réinitialisé au pas suivant, on coupe
          la propagation récursive de GAE avec `(1 - done_t)`.
        """
        gae = 0.0
        for step in reversed(range(self.ptr)):
            delta = (
                self.rewards[step]
                + gamma * (1.0 - self.terminateds[step]) * self.next_values[step]
                - self.values[step]
            )
            gae = delta + gamma * gae_lambda * (1.0 - self.dones[step]) * gae
            self.advantages[step] = gae

        self.returns[: self.ptr] = self.advantages[: self.ptr] + self.values[: self.ptr]

    def get_minibatches(
        self, batch_size: int
    ) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """Génère des mini-batchs mélangés aléatoirement pour l'optimisation PPO."""
        indices = np.random.permutation(self.ptr)

        states_t = torch.as_tensor(self.states[: self.ptr], device=self.device)
        actions_t = torch.as_tensor(self.actions[: self.ptr], device=self.device)
        old_log_probs_t = torch.as_tensor(self.log_probs[: self.ptr], device=self.device)
        advantages_t = torch.as_tensor(self.advantages[: self.ptr], device=self.device)
        returns_t = torch.as_tensor(self.returns[: self.ptr], device=self.device)
        old_values_t = torch.as_tensor(self.values[: self.ptr], device=self.device)

        # Normalisation des avantages sur le rollout complet (stabilise fortement PPO)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        for start_idx in range(0, self.ptr, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            yield (
                states_t[batch_idx],
                actions_t[batch_idx],
                old_log_probs_t[batch_idx],
                advantages_t[batch_idx],
                returns_t[batch_idx],
                old_values_t[batch_idx],
            )


class PPOAgent:
    """
    Agent PPO complet pour le contrôle continu (Pendulum-v1 et TrackMania TMRL).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_dim: int = 64,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        buffer_size: int = 2048,
        normalize_states: bool = True,
        reward_scale: float = 0.1,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.max_action = float(np.max(np.abs(self.action_high)))

        # Hyperparamètres PPO
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.normalize_states = normalize_states
        self.reward_scale = reward_scale

        # Normalisation en ligne des observations
        self.state_rms = RunningMeanStd(shape=(state_dim,))

        # Réseaux Acteur et Critique (Module 2)
        self.actor = ContinuousActor(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=self.max_action,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.critic = Critic(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # Optimiseurs Adam
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)

        # Mémoire tampon On-Policy
        self.buffer = RolloutBuffer(
            buffer_size=buffer_size,
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device,
        )

    def preprocess_state(self, state: np.ndarray, update_stats: bool = False) -> np.ndarray:
        """Normalise l'état si l'option `normalize_states` est activée."""
        state_arr = np.asarray(state, dtype=np.float32).ravel()
        if self.normalize_states:
            if update_stats:
                self.state_rms.update(state_arr[np.newaxis, :])
            return self.state_rms.normalize(state_arr)
        return state_arr

    @torch.no_grad()
    def select_action(
        self, state: np.ndarray, deterministic: bool = False, update_stats: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Sélectionne une action à partir de l'état courant.

        Returns:
            env_action: Action clippée dans [action_low, action_high] à envoyer à Gymnasium.
            raw_action: Action brute échantillonnée de la Gaussienne (stockée dans le buffer).
            log_prob: Log-probabilité log pi_theta(raw_action | s).
            value: Estimation V_phi(s) du Critique.
        """
        norm_state = self.preprocess_state(state, update_stats=update_stats)
        state_tensor = torch.as_tensor(norm_state, dtype=torch.float32, device=self.device).unsqueeze(0)

        action_tensor, log_prob_tensor = self.actor.get_action_and_log_prob(
            state_tensor, deterministic=deterministic
        )
        value_tensor = self.critic(state_tensor)

        raw_action = action_tensor.squeeze(0).cpu().numpy()
        # On clippe l'action envoyée à l'environnement pour respecter ses bornes physiques
        env_action = np.clip(raw_action, self.action_low, self.action_high)
        log_prob = float(log_prob_tensor.item())
        value = float(value_tensor.item())

        return env_action, raw_action, log_prob, value

    @torch.no_grad()
    def get_value(self, state: np.ndarray) -> float:
        """Calcule uniquement V_phi(s) (utile pour bootstrapper le dernier état)."""
        norm_state = self.preprocess_state(state, update_stats=False)
        state_tensor = torch.as_tensor(norm_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.critic(state_tensor).item())

    def update(self) -> Dict[str, float]:
        """
        Exécute la mise à jour PPO sur les données collectées dans le RolloutBuffer :
        1. Calcule les avantages GAE et les retours cibles R_t.
        2. Sur `ppo_epochs` époques, découpe le buffer en mini-batchs et optimise :
           - L'Acteur via l'objectif clippé L^{CLIP} + bonus d'entropie
           - Le Critique via l'erreur quadratique moyenne (MSE) entre V_phi(s) et R_t
        3. Vide le RolloutBuffer.
        """
        self.buffer.compute_returns_and_advantages(
            gamma=self.gamma, gae_lambda=self.gae_lambda
        )

        actor_losses = []
        critic_losses = []
        entropy_losses = []
        approx_kls = []
        clip_fractions = []

        for _ in range(self.ppo_epochs):
            for (
                states,
                actions,
                old_log_probs,
                advantages,
                returns,
                old_values,
            ) in self.buffer.get_minibatches(self.batch_size):

                # 1. Évaluation des actions sous la politique actuelle pi_theta
                new_log_probs, entropy = self.actor.evaluate_actions(states, actions)
                new_values = self.critic(states)

                # 2. Ratio de probabilité : r_t(theta) = pi_theta(a|s) / pi_theta_old(a|s)
                log_ratio = new_log_probs - old_log_probs
                ratio = torch.exp(log_ratio)

                # 3. Objectif clippé PPO : L^{CLIP}
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Bonus d'entropie pour favoriser l'exploration
                entropy_loss = entropy.mean()
                total_actor_loss = actor_loss - self.entropy_coef * entropy_loss

                # 4. Perte du Critique : MSE(V_phi(s), R_t)
                critic_loss = nn.functional.mse_loss(new_values, returns)

                # 5. Rétropropagation de l'Acteur
                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # 6. Rétropropagation du Critique
                self.critic_optimizer.zero_grad()
                (self.value_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # Métriques de diagnostic (pour TensorBoard et le suivi)
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    clip_frac = (torch.abs(ratio - 1.0) > self.clip_eps).float().mean().item()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_losses.append(entropy_loss.item())
                approx_kls.append(approx_kl)
                clip_fractions.append(clip_frac)

        # On vide le buffer On-Policy après la mise à jour
        self.buffer.clear()

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropy_losses)),
            "approx_kl": float(np.mean(approx_kls)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "policy_std": float(torch.exp(self.actor.log_std).mean().item()),
        }

    def save(self, filepath: str) -> None:
        """Sauvegarde les poids de l'Acteur, du Critique et les statistiques de normalisation."""
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "state_rms": self.state_rms.state_dict(),
            "normalize_states": self.normalize_states,
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath: str) -> None:
        """Charge un checkpoint sauvegardé."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic_optimizer" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if "state_rms" in checkpoint:
            self.state_rms.load_state_dict(checkpoint["state_rms"])
        if "normalize_states" in checkpoint:
            self.normalize_states = checkpoint["normalize_states"]
