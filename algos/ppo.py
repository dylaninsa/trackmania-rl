"""Proximal Policy Optimization with a clipped objective and GAE (Schulman et al., 2017).

PPO is on-policy: the agent collects a fixed number of transitions with its current policy, optimizes on
them for a few epochs, then discards them.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks.ppo import ContinuousActor, Critic


class RunningMeanStd:
    """Running mean and variance of the observations, updated incrementally (parallel Welford algorithm).

    Args:
        shape (tuple[int, ...]): Shape of an observation.
        epsilon (float): Initial count, which avoids a division by zero before the first update.
    """

    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        """Adds a batch of observations to the statistics.

        Args:
            x (np.ndarray): Observations of shape ``(N, *shape)``.
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        """Merges the moments of a batch with the current ones.

        Args:
            batch_mean (np.ndarray): Mean of the batch.
            batch_var (np.ndarray): Variance of the batch.
            batch_count (int): Number of observations in the batch.
        """
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = m2 / tot_count
        self.count = tot_count

    def normalize(self, x, clip=10.0, center=False):
        """Scales an observation by the estimated standard deviation, then clips it.

        By default the observation is not centered, so that a physical zero stays at zero, for example
        ``sin(theta) = 0`` when the pendulum is upright.

        Args:
            x (np.ndarray): Observation to normalize.
            clip (float): Bound of the normalized values.
            center (bool): If True, also subtracts the mean.

        Returns:
            np.ndarray: Float32 normalized observation.
        """
        if center:
            norm_x = (x - self.mean) / np.sqrt(self.var + 1e-8)
        else:
            norm_x = x / np.sqrt(self.var + 1e-8)
        return np.clip(norm_x, -clip, clip).astype(np.float32)

    def state_dict(self):
        """Returns the statistics, for checkpoints.

        Returns:
            dict: ``mean``, ``var`` and ``count``.
        """
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state_dict):
        """Restores statistics returned by ``state_dict``.

        Args:
            state_dict (dict): ``mean``, ``var`` and ``count``.
        """
        self.mean = state_dict["mean"]
        self.var = state_dict["var"]
        self.count = state_dict["count"]


class RolloutBuffer:
    """On-policy buffer holding the transitions of one rollout, and their advantages and returns.

    The value of the next state is stored with each transition, so that the advantages are correct even
    when the rollout stops in the middle of an episode.

    Args:
        buffer_size (int): Number of transitions of a rollout.
        state_dim (int): Size of an observation.
        action_dim (int): Size of an action.
        device (torch.device): Device of the tensors returned by ``get_minibatches``.
    """

    def __init__(self, buffer_size, state_dim, action_dim, device):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.clear()

    def clear(self):
        """Empties the buffer."""
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

    def add(self, state, action, log_prob, reward, value, next_value, terminated, truncated):
        """Stores one transition.

        Args:
            state (np.ndarray): Normalized observation before the action.
            action (np.ndarray): Action sampled from the policy, before clipping.
            log_prob (float): Log-probability of the action under the policy that collected it.
            reward (float): Scaled reward.
            value (float): Critic estimate of the state.
            next_value (float): Critic estimate of the next state.
            terminated (bool): Whether the episode terminated after this transition.
            truncated (bool): Whether the episode was truncated after this transition.

        Raises:
            RuntimeError: If the buffer is full.
        """
        if self.ptr >= self.buffer_size:
            raise RuntimeError("The rollout buffer is full: call update() before adding transitions.")
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.next_values[self.ptr] = next_value
        self.terminateds[self.ptr] = float(terminated)
        self.dones[self.ptr] = float(terminated or truncated)
        self.ptr += 1

    def compute_returns_and_advantages(self, gamma=0.99, gae_lambda=0.95):
        """Computes the advantages with GAE (Schulman et al., 2015), and the returns used by the critic.

        ``delta_t = r_t + gamma * (1 - terminated_t) * V(s_{t+1}) - V(s_t)``,
        ``A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}`` and ``R_t = A_t + V(s_t)``.
        After a truncation the next state is still bootstrapped, since the episode could have continued,
        but the recursion is cut, since the next transition belongs to another episode.

        Args:
            gamma (float): Discount factor.
            gae_lambda (float): Trade-off between bias (0) and variance (1) of the advantages.
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

    def get_minibatches(self, batch_size):
        """Yields the stored transitions in shuffled mini-batches.

        The advantages are normalized over the whole rollout.

        Args:
            batch_size (int): Number of transitions of a mini-batch.

        Yields:
            tuple[torch.Tensor, ...]: States, actions, old log-probabilities, advantages, returns and old
                values of the mini-batch.
        """
        indices = np.random.permutation(self.ptr)
        states_t = torch.as_tensor(self.states[: self.ptr], device=self.device)
        actions_t = torch.as_tensor(self.actions[: self.ptr], device=self.device)
        old_log_probs_t = torch.as_tensor(self.log_probs[: self.ptr], device=self.device)
        advantages_t = torch.as_tensor(self.advantages[: self.ptr], device=self.device)
        returns_t = torch.as_tensor(self.returns[: self.ptr], device=self.device)
        old_values_t = torch.as_tensor(self.values[: self.ptr], device=self.device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        for start_idx in range(0, self.ptr, batch_size):
            batch_idx = indices[start_idx: start_idx + batch_size]
            yield (
                states_t[batch_idx],
                actions_t[batch_idx],
                old_log_probs_t[batch_idx],
                advantages_t[batch_idx],
                returns_t[batch_idx],
                old_values_t[batch_idx],
            )


class PPOAgent:
    """PPO agent with a Gaussian actor, a critic, an observation normalizer and its rollout buffer.

    The agent works with actions in [-1, 1], like SAC: the training loop rescales them to the bounds of the
    environment.

    Args:
        obs_dim (int): Size of an observation.
        act_dim (int): Size of an action.
        config (dict): Hyperparameters ``hidden``, ``lr_actor``, ``lr_critic``, ``gamma``, ``gae_lambda``,
            ``clip_eps``, ``entropy_coef``, ``value_coef``, ``max_grad_norm``, ``ppo_epochs``, ``batch_size``,
            ``buffer_size``, ``normalize_states``, ``reward_scale`` and ``device``.
    """

    def __init__(self, obs_dim, act_dim, config):
        self.device = torch.device(config["device"])
        self.gamma = config["gamma"]
        self.gae_lambda = config["gae_lambda"]
        self.clip_eps = config["clip_eps"]
        self.entropy_coef = config["entropy_coef"]
        self.value_coef = config["value_coef"]
        self.max_grad_norm = config["max_grad_norm"]
        self.ppo_epochs = config["ppo_epochs"]
        self.batch_size = config["batch_size"]
        self.normalize_states = config["normalize_states"]
        self.reward_scale = config["reward_scale"]
        self.lr_actor = config["lr_actor"]
        self.lr_critic = config["lr_critic"]

        self.state_rms = RunningMeanStd(shape=(obs_dim,))
        self.actor = ContinuousActor(obs_dim, act_dim, max_action=1.0, hidden_dim=config["hidden"])
        self.actor.to(self.device)
        self.critic = Critic(obs_dim, hidden_dim=config["hidden"]).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic, eps=1e-5)
        self.buffer = RolloutBuffer(config["buffer_size"], obs_dim, act_dim, self.device)

    def act(self, obs, deterministic=False):
        """Chooses an action for one observation, without updating the normalization statistics.

        Args:
            obs (np.ndarray): Raw observation.
            deterministic (bool): If True, returns the mean action instead of a sample, for evaluation.

        Returns:
            np.ndarray: Float32 action of shape ``(act_dim,)``, in [-1, 1].
        """
        return self.select_action(obs, deterministic=deterministic)[0]

    def reset(self):
        """Does nothing, because the agent has no state within an episode; it only exists for the common
        interface with the baseline policies."""

    def store(self, obs, raw_action, log_prob, reward, value, next_obs, terminated, truncated):
        """Normalizes and stores one transition in the rollout buffer.

        Args:
            obs (np.ndarray): Raw observation before the action.
            raw_action (np.ndarray): Action sampled from the Gaussian, as returned by ``select_action``.
            log_prob (float): Log-probability returned by ``select_action``.
            reward (float): Reward of the environment, before scaling.
            value (float): Critic estimate returned by ``select_action``.
            next_obs (np.ndarray): Raw observation after the action.
            terminated (bool): Whether the episode terminated after this transition.
            truncated (bool): Whether the episode was truncated after this transition.
        """
        self.buffer.add(
            state=self.preprocess_state(obs),
            action=raw_action,
            log_prob=log_prob,
            reward=reward * self.reward_scale,
            value=value,
            next_value=self.get_value(next_obs),
            terminated=terminated,
            truncated=truncated,
        )

    def set_learning_rate_fraction(self, fraction):
        """Sets both learning rates to a fraction of their initial value, for a linear decay.

        Args:
            fraction (float): Fraction of the learning rates given in the configuration.
        """
        for group in self.actor_optimizer.param_groups:
            group["lr"] = self.lr_actor * fraction
        for group in self.critic_optimizer.param_groups:
            group["lr"] = self.lr_critic * fraction

    def preprocess_state(self, state, update_stats=False):
        """Flattens an observation and normalizes it if ``normalize_states`` is set.

        Args:
            state (np.ndarray): Raw observation.
            update_stats (bool): Whether to add the observation to the running statistics first.

        Returns:
            np.ndarray: Float32 observation of shape ``(obs_dim,)``.
        """
        state_arr = np.asarray(state, dtype=np.float32).ravel()
        if self.normalize_states:
            if update_stats:
                self.state_rms.update(state_arr[np.newaxis, :])
            return self.state_rms.normalize(state_arr)
        return state_arr

    @torch.no_grad()
    def select_action(self, state, deterministic=False, update_stats=False):
        """Chooses an action for one observation, without computing gradients.

        Args:
            state (np.ndarray): Raw observation.
            deterministic (bool): If True, returns the mean action instead of a sample, for evaluation.
            update_stats (bool): Whether to add the observation to the normalization statistics.

        Returns:
            tuple[np.ndarray, np.ndarray, float, float]: Action clipped to [-1, 1],
                raw action sampled from the Gaussian (the one stored in the buffer), its log-probability, and
                the critic estimate of the state.
        """
        norm_state = self.preprocess_state(state, update_stats=update_stats)
        state_tensor = torch.as_tensor(norm_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_tensor, log_prob_tensor = self.actor.get_action_and_log_prob(state_tensor, deterministic)
        value_tensor = self.critic(state_tensor)

        raw_action = action_tensor.squeeze(0).cpu().numpy()
        env_action = np.clip(raw_action, -1.0, 1.0)
        return env_action, raw_action, float(log_prob_tensor.item()), float(value_tensor.item())

    @torch.no_grad()
    def get_value(self, state):
        """Computes the critic estimate of one observation.

        Args:
            state (np.ndarray): Raw observation.

        Returns:
            float: Estimate of V(s).
        """
        norm_state = self.preprocess_state(state, update_stats=False)
        state_tensor = torch.as_tensor(norm_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.critic(state_tensor).item())

    def update(self):
        """Optimizes the actor and the critic on the current rollout, then empties the buffer.

        For ``ppo_epochs`` passes over shuffled mini-batches, the actor minimizes the clipped surrogate loss
        ``-min(ratio * A, clip(ratio, 1 - eps, 1 + eps) * A)`` minus the entropy bonus, and the critic
        minimizes the mean squared error between ``V(s)`` and the GAE returns. Each network has its own
        optimizer and its gradients are clipped to ``max_grad_norm``.

        Returns:
            dict: ``actor_loss``, ``critic_loss``, ``entropy``, ``approx_kl``, ``clip_fraction`` and
                ``policy_std``, averaged over the mini-batches, as floats.
        """
        self.buffer.compute_returns_and_advantages(gamma=self.gamma, gae_lambda=self.gae_lambda)

        actor_losses = []
        critic_losses = []
        entropy_losses = []
        approx_kls = []
        clip_fractions = []

        for _ in range(self.ppo_epochs):
            for states, actions, old_log_probs, advantages, returns, old_values in \
                    self.buffer.get_minibatches(self.batch_size):
                new_log_probs, entropy = self.actor.evaluate_actions(states, actions)
                new_values = self.critic(states)

                log_ratio = new_log_probs - old_log_probs
                ratio = torch.exp(log_ratio)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = entropy.mean()
                total_actor_loss = actor_loss - self.entropy_coef * entropy_loss
                critic_loss = nn.functional.mse_loss(new_values, returns)

                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                (self.value_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    clip_frac = (torch.abs(ratio - 1.0) > self.clip_eps).float().mean().item()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_losses.append(entropy_loss.item())
                approx_kls.append(approx_kl)
                clip_fractions.append(clip_frac)

        self.buffer.clear()

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropy_losses)),
            "approx_kl": float(np.mean(approx_kls)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "policy_std": float(torch.exp(self.actor.log_std).mean().item()),
        }

    def save(self, filepath):
        """Saves the networks, the optimizers and the normalization statistics to a single file.

        Args:
            filepath (str | pathlib.Path): Destination ``.pt`` file.
        """
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "state_rms": self.state_rms.state_dict(),
            "normalize_states": self.normalize_states,
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath):
        """Restores an agent saved with ``save``.

        Args:
            filepath (str | pathlib.Path): ``.pt`` file written by ``save``.
        """
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
