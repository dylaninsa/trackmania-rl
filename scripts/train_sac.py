"""Trains SAC on Pendulum-v1 or on TrackMania.

The agent plays whole episodes and stores every transition in a replay buffer. During the first
``start_steps`` steps it plays uniformly random actions, to fill the buffer with varied data. At the end of
each later episode it performs ``updates_per_step`` gradient steps per step of the episode, each on a
mini-batch sampled from the buffer, and it is evaluated with its deterministic policy once ``eval_every``
more steps have been played. No computation interrupts an episode, which matters in TrackMania, where the
game keeps running between two steps.

The run stops after ``total_steps`` steps, or ``--max-steps``, even in the middle of an episode; a last
evaluation then follows, unless one has just taken place. The logs go to TensorBoard in ``runs/``. After each
evaluation, the evaluations are written to ``results/<config>_seed<seed>.csv``, the agent to
``checkpoints/<config>_seed<seed>.pt``, and also to ``checkpoints/<config>_seed<seed>_best.pt`` when it beats
every previous evaluation (mean distance in TrackMania, mean return otherwise).

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
from training.common import EvaluationRecorder, make_envs, set_seed


def main():
    """Trains the agent described by the configuration given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="fichier JSON des hyperparamètres")
    parser.add_argument("--seed", type=int, default=None, help="remplace la graine de la configuration")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="arrête plus tôt, sans changer total_steps ni la décroissance du taux d'apprentissage")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    seed = config["seed"] if args.seed is None else args.seed
    name = f"{Path(args.config).stem}_seed{seed}"
    set_seed(seed)
    rng = np.random.default_rng(seed)

    env, eval_env = make_envs(config)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = SACAgent(obs_dim, act_dim, config)
    buffer = ReplayBuffer(obs_dim, act_dim, config["buffer_size"], device=config["device"], seed=seed)
    writer = SummaryWriter(f"runs/{name}")
    recorder = EvaluationRecorder(name, writer)

    obs, _ = env.reset(seed=seed)
    agent.reset()
    episode_return = 0.0
    episode_steps = 0
    episode = 0
    step = 0
    next_eval = config["eval_every"]
    max_steps = config["total_steps"] if args.max_steps is None else args.max_steps
    while step < max_steps:
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
            recorder.evaluate(agent, eval_env, config["eval_episodes"], seed + 1000, step, episode)
            next_eval += config["eval_every"]

        obs, _ = env.reset()
        agent.reset()
        episode_return = 0.0
        episode_steps = 0

    if not recorder.evaluations or recorder.evaluations[-1]["step"] < step:
        recorder.evaluate(agent, eval_env, config["eval_episodes"], seed + 1000, step, episode)
    env.close()
    if eval_env is not env:
        eval_env.close()
    writer.close()
    print(f"Évaluations dans {recorder.results_file}, agents dans {recorder.checkpoint_file} et {recorder.best_file}")


if __name__ == "__main__":
    main()
