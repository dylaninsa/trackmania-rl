"""Trains SAC on Pendulum-v1 or on TrackMania.

The agent plays whole episodes and stores every transition in a replay buffer. During the first
``start_steps`` steps it plays uniformly random actions, to fill the buffer with varied data. At the end of
each later episode it performs ``updates_per_step`` gradient steps per step of the episode, each on a
mini-batch sampled from the buffer, and it is evaluated with its deterministic policy once ``eval_every``
more steps have been played. No computation interrupts an episode, which matters in TrackMania, where the
game keeps running between two steps.

The logs go to TensorBoard in ``runs/``. After each evaluation, the evaluations are written to
``results/<config>_seed<seed>.csv`` and the agent to ``checkpoints/<config>_seed<seed>.pt``.

Examples:
    Train on Pendulum with the seed of the configuration::

        uv run python scripts/train_sac.py --config configs/sac_pendulum.json

    Train on TrackMania, with the game open on the track::

        uv run python scripts/train_sac.py --config configs/sac_trackmania.json --seed 1
"""

import argparse
import json
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from algos.replay_buffer import ReplayBuffer
from algos.sac import SACAgent
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
    set_seed(seed)
    rng = np.random.default_rng(seed)

    env, eval_env = make_envs(config)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = SACAgent(obs_dim, act_dim, config)
    buffer = ReplayBuffer(obs_dim, act_dim, config["buffer_size"], device=config["device"], seed=seed)
    writer = SummaryWriter(f"runs/{name}")
    evaluations = []

    obs, _ = env.reset(seed=seed)
    agent.reset()
    episode_return = 0.0
    episode_steps = 0
    episode = 0
    step = 0
    next_eval = config["eval_every"]
    while step < config["total_steps"]:
        if step < config["start_steps"]:
            action = rng.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
        else:
            action = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action * env.action_space.high)
        buffer.add(obs, action, reward, next_obs, terminated)
        step += 1
        episode_steps += 1
        episode_return += float(reward)
        obs = next_obs
        if not (terminated or truncated):
            continue

        episode += 1
        writer.add_scalar("train/episode_return", episode_return, step)

        if step >= config["start_steps"]:
            metrics = []
            for _ in range(int(episode_steps * config["updates_per_step"])):
                metrics.append(agent.update(buffer.sample(config["batch_size"])))
            for key in metrics[0]:
                writer.add_scalar(f"train/{key}", float(np.mean([m[key] for m in metrics])), step)

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
        episode_steps = 0

    agent.save(checkpoint_file)
    env.close()
    if eval_env is not env:
        eval_env.close()
    writer.close()
    print(f"Évaluations dans {results_file}, agent dans {checkpoint_file}")


if __name__ == "__main__":
    main()
