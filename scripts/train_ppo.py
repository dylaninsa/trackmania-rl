"""Trains PPO on a Gymnasium environment with continuous actions, such as Pendulum-v1.

The agent is updated every ``buffer_size`` steps on the rollout it has just collected, and evaluated every
``eval_every`` steps with its deterministic policy on a separate environment. The logs go to TensorBoard in
``runs/``, the evaluations to ``results/<config>_seed<seed>.csv`` and the final agent to
``checkpoints/<config>_seed<seed>.pt``.

Examples:
    Train with the seed of the configuration::

        uv run python scripts/train_ppo.py --config configs/ppo_pendulum.json

    Train with another seed::

        uv run python scripts/train_ppo.py --config configs/ppo_pendulum.json --seed 1
"""

import argparse
import csv
import json
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from algos.ppo import PPOAgent


def set_seed(seed):
    """Seeds the random generators of Python, NumPy and PyTorch.

    Args:
        seed (int): Seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(agent, env, episodes, seed):
    """Plays several episodes with the deterministic policy of the agent.

    Args:
        agent (PPOAgent): Agent to evaluate.
        env (gymnasium.Env): Evaluation environment, separate from the training one.
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


def main():
    """Trains the agent described by the configuration given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="fichier JSON des hyperparamètres")
    parser.add_argument("--seed", type=int, default=None, help="remplace la graine de la configuration")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    seed = config["seed"] if args.seed is None else args.seed
    name = f"{Path(args.config).stem}_seed{seed}"
    set_seed(seed)

    env = gym.make(config["env"])
    eval_env = gym.make(config["env"])
    env.action_space.seed(seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = PPOAgent(obs_dim, act_dim, config)
    writer = SummaryWriter(f"runs/{name}")
    evaluations = []

    obs, _ = env.reset(seed=seed)
    episode_return = 0.0
    for step in range(1, config["total_steps"] + 1):
        action, raw_action, log_prob, value = agent.select_action(obs, update_stats=True)
        next_obs, reward, terminated, truncated, _ = env.step(action * env.action_space.high)
        agent.store(obs, raw_action, log_prob, reward, value, next_obs, terminated, truncated)
        episode_return += float(reward)
        obs = next_obs
        if terminated or truncated:
            writer.add_scalar("train/episode_return", episode_return, step)
            obs, _ = env.reset()
            episode_return = 0.0

        if step % config["buffer_size"] == 0:
            fraction = max(1.0 - step / config["total_steps"], config["min_lr_fraction"])
            agent.set_learning_rate_fraction(fraction)
            for key, metric in agent.update().items():
                writer.add_scalar(f"train/{key}", metric, step)

        if step % config["eval_every"] == 0:
            mean, std = evaluate(agent, eval_env, config["eval_episodes"], seed + 1000)
            evaluations.append({"step": step, "mean_return": mean, "std_return": std})
            writer.add_scalar("eval/mean_return", mean, step)
            print(f"Pas {step:7d} : retour en évaluation {mean:8.1f} ± {std:6.1f}")

    env.close()
    eval_env.close()
    writer.close()

    results_file = Path("results") / f"{name}.csv"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.DictWriter(f, fieldnames=["step", "mean_return", "std_return"])
        csv_writer.writeheader()
        csv_writer.writerows(evaluations)

    checkpoint_file = Path("checkpoints") / f"{name}.pt"
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    agent.save(checkpoint_file)
    print(f"Évaluations dans {results_file}, agent dans {checkpoint_file}")


if __name__ == "__main__":
    main()
