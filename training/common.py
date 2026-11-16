"""Pieces shared by the training scripts: seeding, environments, evaluation and results files."""

import csv
import math
import random
from pathlib import Path

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
        dict: ``mean_return`` and ``std_return``. In TrackMania, whose ``info`` holds ``distance_m`` and
            ``end_of_track``, also ``mean_distance_m`` and ``finish_rate``, read at the last step of each
            episode.
    """
    returns = []
    distances = []
    finished = []
    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        agent.reset()
        episode_return = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action * env.action_space.high)
            episode_return += float(reward)
        returns.append(episode_return)
        if "distance_m" in info:
            distances.append(info["distance_m"])
            finished.append(float(info["end_of_track"]))
    result = {"mean_return": float(np.mean(returns)), "std_return": float(np.std(returns))}
    if distances:
        result["mean_distance_m"] = float(np.mean(distances))
        result["finish_rate"] = float(np.mean(finished))
    return result


def save_evaluations(evaluations, path):
    """Writes the evaluations to a CSV file, replacing its previous content.

    Args:
        evaluations (list[dict]): ``step`` and the results of ``evaluate`` for each evaluation. The columns
            are the keys of the first one.
        path (pathlib.Path): Destination file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.DictWriter(f, fieldnames=list(evaluations[0]))
        csv_writer.writeheader()
        csv_writer.writerows(evaluations)


def score(result):
    """Number used to pick the best checkpoint of a run.

    Args:
        result (dict): Result of ``evaluate``.

    Returns:
        float: Mean distance in TrackMania, mean return otherwise.
    """
    return result.get("mean_distance_m", result["mean_return"])


class EvaluationRecorder:
    """Evaluates the agent during a run and keeps its results, its last checkpoint and its best checkpoint.

    Args:
        name (str): Name of the run, ``<config>_seed<seed>``.
        writer (SummaryWriter): TensorBoard writer of the run.
    """

    def __init__(self, name, writer):
        self.writer = writer
        self.results_file = Path("results") / f"{name}.csv"
        self.checkpoint_file = Path("checkpoints") / f"{name}.pt"
        self.best_file = Path("checkpoints") / f"{name}_best.pt"
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        self.evaluations = []
        self.best_score = -math.inf

    def evaluate(self, agent, env, episodes, seed, step, episode):
        """Evaluates the agent, logs and saves the results, and saves the agent.

        The agent is always saved to ``checkpoint_file``, and also to ``best_file`` when its score is
        strictly higher than every previous one.

        Args:
            agent (PPOAgent | SACAgent): Agent to evaluate.
            env (gymnasium.Env): Evaluation environment.
            episodes (int): Number of episodes.
            seed (int): Seed of the first episode.
            step (int): Number of training steps played so far.
            episode (int): Number of training episodes finished so far.
        """
        result = evaluate(agent, env, episodes, seed)
        self.evaluations.append({"step": step, **result})
        for key, value in result.items():
            self.writer.add_scalar(f"eval/{key}", value, step)
        save_evaluations(self.evaluations, self.results_file)
        agent.save(self.checkpoint_file)
        if score(result) > self.best_score:
            self.best_score = score(result)
            agent.save(self.best_file)
        summary = ", ".join(f"{key} {value:.2f}" for key, value in result.items())
        print(f"Pas {step:7d}, épisode {episode:5d} : {summary}", flush=True)
