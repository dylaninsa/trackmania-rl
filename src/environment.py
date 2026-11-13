"""
Module 1 : L'environnement (Interface Gymnasium pour Pendulum-v1 et TMRL).

Ce module abstrait l'environnement pour que les modules 2 (Réseaux), 3 (PPO) et
4 (Boucle d'entraînement) fonctionnent de manière 100 % identique sur :
- Pendulum-v1 : état de dimension 3, action continue de dimension 1 dans [-2.0, 2.0]
- TMRL (LIDAR) : état aplati de dimension 83 (vitesse + 4x19 LIDAR + 2x3 actions),
                 action continue de dimension 3 dans [-1.0, 1.0] (gaz, frein, direction)
"""

from typing import Optional, Tuple, Dict, Any
import gymnasium as gym
import numpy as np


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


class FlattenTupleObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper destiné à TMRL (pour la suite du projet) :
    TMRL renvoie une observation sous forme de Tuple :
      (vitesse [1], 4 balayages LIDAR [4, 19], action_t-1 [3], action_t-2 [3])
    Ce wrapper concatène automatiquement tous les éléments du tuple en un seul
    vecteur 1D de type float32 (dimension 83 sur TMRL), exactement comme Pendulum-v1
    qui fournit un vecteur 1D de dimension 3.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        sample_obs, _ = env.reset()
        flat_sample = self._flatten(sample_obs)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=flat_sample.shape,
            dtype=np.float32,
        )

    @staticmethod
    def _flatten(obs: Any) -> np.ndarray:
        if isinstance(obs, (tuple, list)):
            parts = [np.asarray(item, dtype=np.float32).ravel() for item in obs]
            return np.concatenate(parts, axis=0)
        return np.asarray(obs, dtype=np.float32).ravel()

    def observation(self, observation: Any) -> np.ndarray:
        return self._flatten(observation)


def make_env(
    env_id: str = "Pendulum-v1",
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
) -> gym.Env:
    """
    Fabrique et retourne l'environnement Gymnasium standardisé.

    Args:
        env_id: Identifiant de l'environnement ("Pendulum-v1" ou "TMRL" dans le 2e temps).
        render_mode: None pour l'entraînement rapide, "human" pour visualiser.
        seed: Graine aléatoire pour la reproductibilité.

    Returns:
        env: Instance de gym.Env avec observation 1D (float32) et action continue Box.
    """
    if env_id.upper().startswith("TMRL"):
        # --- PRÉPARATION POUR LE SECOND TEMPS (TrackMania via TMRL) ---
        try:
            import tmrl
            env = tmrl.get_environment()
            env = FlattenTupleObservationWrapper(env)
        except ImportError as exc:
            raise ImportError(
                "Le package 'tmrl' n'est pas installé dans cet environnement. "
                "Utilisez env_id='Pendulum-v1' pour la validation algorithmique."
            ) from exc
    else:
        # --- ENVIRONNEMENT DE VALIDATION ALGORITHMIQUE (Pendulum-v1) ---
        env = gym.make(env_id, render_mode=render_mode)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    return env


def get_env_specs(env: gym.Env) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """
    Extrait les dimensions et les bornes d'action de n'importe quel environnement continu.

    Returns:
        state_dim: Dimension du vecteur d'observation (3 pour Pendulum, 83 pour TMRL LIDAR)
        action_dim: Dimension du vecteur d'action (1 pour Pendulum, 3 pour TMRL)
        action_low: Borne inférieure des actions ([-2.0] pour Pendulum, [-1, -1, -1] pour TMRL)
        action_high: Borne supérieure des actions ([+2.0] pour Pendulum, [+1, +1, +1] pour TMRL)
    """
    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    return state_dim, action_dim, action_low, action_high
