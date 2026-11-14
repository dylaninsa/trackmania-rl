"""Pieces shared by the training scripts: seeding, environments, evaluation and results files."""

import csv
import random

import gymnasium as gym
import numpy as np
import torch


def set_seed(seed):
    """Seeds the random generators of Python, NumPy and PyTorch.

    Args:
        seed (int): Seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_envs(config):
    """Creates the training and evaluation environments.

    ``envs.trackmania`` is imported only for TrackMania, so that the other environments do not need TMRL.

    Args:
        config (dict): Configuration, with ``env`` and ``ep_max_length``.

    Returns:
        tuple[gymnasium.Env, gymnasium.Env]: Training and evaluation environments. They are the same object
            for TrackMania, since only one game can be driven.
    """
    if config["env"] == "trackmania":
        from envs.trackmania import make_env
        env = make_env(ep_max_length=config["ep_max_length"])
        return env, env
    return gym.make(config["env"]), gym.make(config["env"])


def evaluate(agent, env, episodes, seed):
    """Plays several episodes with the deterministic policy of the agent.

    Args:
        agent (PPOAgent | SACAgent): Agent to evaluate.
        env (gymnasium.Env): Evaluation environment.
        episodes (int): Number of episodes.
        seed (int): Seed of the first episode; episode ``k`` uses ``seed + k``.

    Returns:
        tuple[float, float]: Mean and standard deviation of the returns.
    """
    returns = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        agent.reset()
        episode_return = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action * env.action_space.high)
            episode_return += float(reward)
        returns.append(episode_return)
    return float(np.mean(returns)), float(np.std(returns))


def save_evaluations(evaluations, path):
    """Writes the evaluations to a CSV file, replacing its previous content.

    Args:
        evaluations (list[dict]): ``step``, ``mean_return`` and ``std_return`` of each evaluation.
        path (pathlib.Path): Destination file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.DictWriter(f, fieldnames=["step", "mean_return", "std_return"])
        csv_writer.writeheader()
        csv_writer.writerows(evaluations)
