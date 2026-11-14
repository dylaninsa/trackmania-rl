"""Trains PPO on Pendulum-v1 or on TrackMania.

The agent plays whole episodes: at the end of each one, it is updated if it has collected at least
``buffer_size`` transitions, and evaluated with its deterministic policy once ``eval_every`` more steps have
been played. No computation interrupts an episode, which matters in TrackMania, where the game keeps running
between two steps. For TrackMania (``"env": "trackmania"``) the evaluation uses the training environment,
since only one game is open; otherwise it uses a separate environment.

The logs go to TensorBoard in ``runs/``. After each evaluation, the evaluations are written to
``results/<config>_seed<seed>.csv`` and the agent to ``checkpoints/<config>_seed<seed>.pt``.

Examples:
    Train on Pendulum with the seed of the configuration::

        uv run python scripts/train_ppo.py --config configs/ppo_pendulum.json

    Train on TrackMania, with the game open on the track::

        uv run python scripts/train_ppo.py --config configs/ppo_trackmania.json --seed 1
"""

import argparse
import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from algos.ppo import PPOAgent
from training.common import evaluate, make_envs, save_evaluations, set_seed


def main():
    """Trains the agent described by the configuration given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="fichier JSON des hyperparamètres")
    parser.add_argument("--seed", type=int, default=None, help="remplace la graine de la configuration")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    seed = config["seed"] if args.seed is None else args.seed
    name = f"{Path(args.config).stem}_seed{seed}"
    results_file = Path("results") / f"{name}.csv"
    checkpoint_file = Path("checkpoints") / f"{name}.pt"
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    env, eval_env = make_envs(config)
    env.action_space.seed(seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = PPOAgent(obs_dim, act_dim, config)
    writer = SummaryWriter(f"runs/{name}")
    evaluations = []

    obs, _ = env.reset(seed=seed)
    agent.reset()
    episode_return = 0.0
    episode = 0
    step = 0
    next_eval = config["eval_every"]
    while step < config["total_steps"]:
        action, raw_action, log_prob, value = agent.select_action(obs, update_stats=True)
        next_obs, reward, terminated, truncated, _ = env.step(action * env.action_space.high)
        agent.store(obs, raw_action, log_prob, reward, value, next_obs, terminated, truncated)
        step += 1
        episode_return += float(reward)
        obs = next_obs
        if not (terminated or truncated):
            continue

        episode += 1
        writer.add_scalar("train/episode_return", episode_return, step)

        if agent.ready_to_update():
            fraction = max(1.0 - step / config["total_steps"], config["min_lr_fraction"])
            agent.set_learning_rate_fraction(fraction)
            for key, metric in agent.update().items():
                writer.add_scalar(f"train/{key}", metric, step)

        if step >= next_eval:
            mean, std = evaluate(agent, eval_env, config["eval_episodes"], seed + 1000)
            evaluations.append({"step": step, "mean_return": mean, "std_return": std})
            writer.add_scalar("eval/mean_return", mean, step)
            save_evaluations(evaluations, results_file)
            agent.save(checkpoint_file)
            print(f"Pas {step:7d}, épisode {episode:5d} : retour en évaluation {mean:8.1f} ± {std:6.1f}")
            next_eval += config["eval_every"]

        obs, _ = env.reset()
        agent.reset()
        episode_return = 0.0

    agent.save(checkpoint_file)
    env.close()
    if eval_env is not env:
        eval_env.close()
    writer.close()
    print(f"Évaluations dans {results_file}, agent dans {checkpoint_file}")


if __name__ == "__main__":
    main()
