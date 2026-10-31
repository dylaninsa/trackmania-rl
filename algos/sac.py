"""Soft Actor-Critic with automatic entropy tuning (Haarnoja et al., 2018).

The agent works with any observation and action sizes, and with actions in [-1, 1]. It does not own the
replay buffer: the training loop samples batches and passes them to ``update``.
"""

import copy
from pathlib import Path

import torch
import torch.nn.functional as F

from networks.sac import Actor, Critic


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
        self.device = torch.device(config["device"])
        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.target_entropy = -float(act_dim)

        self.actor = Actor(obs_dim, act_dim, config["hidden"]).to(self.device)
        self.q1 = Critic(obs_dim, act_dim, config["hidden"]).to(self.device)
        self.q2 = Critic(obs_dim, act_dim, config["hidden"]).to(self.device)
        self.q1_target = copy.deepcopy(self.q1).requires_grad_(False)
        self.q2_target = copy.deepcopy(self.q2).requires_grad_(False)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config["lr"])
        self.critic_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()),
                                                 lr=config["lr"])
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config["lr"])

    def act(self, obs, deterministic=False):
        """Chooses an action for one observation, without computing gradients.

        Args:
            obs (np.ndarray): Observation of shape ``(obs_dim,)``.
            deterministic (bool): If True, returns the mean action instead of a sample, for evaluation.

        Returns:
            np.ndarray: Float32 action of shape ``(act_dim,)``, in [-1, 1].
        """
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
            action, _ = self.actor(obs_tensor, deterministic)
            return action.cpu().numpy()[0]

    def reset(self):
        """Does nothing, because the agent has no state within an episode; it only exists for the common
        interface with the baseline policies."""
    
    def compute_target(self, batch):
        """Computes the soft Bellman target of the critics.

        ``y = rew + gamma * (1 - terminated) * (min(Q1_target, Q2_target)(s', a') - alpha * log pi(a'|s'))``,
        where ``a'`` is sampled from the current actor, since SAC has no target actor. Taking the minimum of
        the two target critics limits the overestimation of Q. The ``(1 - terminated)`` mask removes the
        future after a real end of episode, but not after a truncation. Everything is computed without
        gradients: the target is a fixed value that the critics must reach, not something they can move.

        Args:
            batch (dict): Batch in the format returned by ``ReplayBuffer.sample``.

        Returns:
            torch.Tensor: Target of shape ``(B,)``.
        """
        with torch.no_grad():
            next_action, next_log_prob = self.actor(batch["next_obs"])
            q1_target = self.q1_target(batch["next_obs"], next_action)
            q2_target = self.q2_target(batch["next_obs"], next_action)
            q_target = torch.min(q1_target, q2_target)
            alpha = self.log_alpha.exp()
            y = batch["rew"] + self.gamma * (1 - batch["terminated"]) * (q_target - alpha * next_log_prob)
            return y

    def update_critics(self, batch):
        """Performs one gradient step on both critics towards the Bellman target.

        The loss is the sum of the mean squared errors of the two critics. Both are updated by the same
        optimizer, since they share the same target.

        Args:
            batch (dict): Batch in the format returned by ``ReplayBuffer.sample``.

        Returns:
            tuple[float, float]: Critic loss, and mean of the minimum of the two critics' estimates on the
                batch, for monitoring.
        """
        y = self.compute_target(batch)
        q1 = self.q1(batch["obs"], batch["act"])
        q2 = self.q2(batch["obs"], batch["act"])

        loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return loss.item(), torch.min(q1, q2).mean().item()

    def update_actor_and_alpha(self, batch):
        """Performs one gradient step on the actor, then one on the temperature.

        The actor draws new actions for the batch observations with the reparameterization trick and
        minimizes ``alpha * log pi(a|s) - min(Q1, Q2)(s, a)``: it looks for actions with a high Q while
        staying as random as possible. The current critics are used, not the targets, and ``alpha`` is
        detached so that the actor cannot change it.

        The temperature minimizes ``-log_alpha * (log pi(a|s) + target_entropy)``: ``alpha`` grows when the
        entropy of the policy falls below the target and shrinks when it is above. The log-probability is
        detached so that this loss does not change the actor. ``log_alpha`` is learned instead of ``alpha``
        to keep ``alpha`` positive.

        The actor loss also leaves gradients on the critics' weights. They are never applied, since only the
        actor's optimizer steps, and ``update_critics`` clears them before its own backward pass.

        Args:
            batch (dict): Batch in the format returned by ``ReplayBuffer.sample``. Only ``obs`` is used.

        Returns:
            tuple[float, float, float, float]: Actor loss, temperature loss, temperature after the update,
                and entropy of the policy estimated by ``-mean(log pi(a|s))``.
        """
        act, log_prob = self.actor(batch["obs"])
        q = torch.min(self.q1(batch["obs"], act), self.q2(batch["obs"], act))
        alpha = self.log_alpha.exp().detach()

        actor_loss = (alpha * log_prob - q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return actor_loss.item(), alpha_loss.item(), self.log_alpha.exp().item(), -log_prob.mean().item()

    def update_targets(self):
        """Moves the target critics slowly towards the critics with a Polyak average.

        Each target parameter becomes ``(1 - tau) * target + tau * critic``. With a small ``tau``, the
        targets follow the critics with a delay, which keeps the Bellman target stable from one update to
        the next. The parameters are modified in place.
        """
        with torch.no_grad():
            for p, p_target in zip(self.q1.parameters(), self.q1_target.parameters()):
                p_target.mul_(1 - self.tau)
                p_target.add_(self.tau * p)

            for p, p_target in zip(self.q2.parameters(), self.q2_target.parameters()):
                p_target.mul_(1 - self.tau)
                p_target.add_(self.tau * p)

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
        update_critic_loss, q_mean = self.update_critics(batch)
        update_actor_loss, update_alpha_loss, alpha, entropy = self.update_actor_and_alpha(batch)
        self.update_targets()
        return {
            "critic_loss": update_critic_loss,
            "actor_loss": update_actor_loss,
            "alpha_loss": update_alpha_loss,
            "alpha": alpha,
            "q_mean": q_mean,
            "entropy": entropy
        }

    def save(self, path):
        """Saves the networks, the targets, the optimizers and the temperature to a single file.

        Args:
            path (str | pathlib.Path): Destination ``.pt`` file.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }, path)

    def load(self, path):
        """Restores an agent saved with ``save``.

        Args:
            path (str | pathlib.Path): ``.pt`` file written by ``save``.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor"])
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
        self.q1_target.load_state_dict(checkpoint["q1_target"])
        self.q2_target.load_state_dict(checkpoint["q2_target"])
        
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        with torch.no_grad():
            self.log_alpha.copy_(checkpoint["log_alpha"])
