"""Trains PPO on Pendulum-v1 or on TrackMania.

The agent plays whole episodes: at the end of each one, it is updated if it has collected at least
``buffer_size`` transitions, and evaluated with its deterministic policy once ``eval_every`` more steps have
been played. No computation interrupts an episode, which matters in TrackMania, where the game keeps running
between two steps. For TrackMania (``"env": "trackmania"``) the evaluation uses the training environment,
since only one game is open; otherwise it uses a separate environment.

The run stops after ``total_steps`` steps, or ``--max-steps``, even in the middle of an episode; a last
evaluation then follows, unless one has just taken place. The logs go to TensorBoard in ``runs/``. After each
evaluation, the evaluations are written to ``results/<config>_seed<seed>.csv``, the agent to
``checkpoints/<config>_seed<seed>.pt``, and also to ``checkpoints/<config>_seed<seed>_best.pt`` when it beats
every previous evaluation (mean distance in TrackMania, mean return otherwise).

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

    env, eval_env = make_envs(config)
    env.action_space.seed(seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = PPOAgent(obs_dim, act_dim, config)
    writer = SummaryWriter(f"runs/{name}")
    recorder = EvaluationRecorder(name, writer)

    obs, _ = env.reset(seed=seed)
    agent.reset()
    episode_return = 0.0
    episode = 0
    step = 0
    next_eval = config["eval_every"]
    max_steps = config["total_steps"] if args.max_steps is None else args.max_steps
    while step < max_steps:
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
            recorder.evaluate(agent, eval_env, config["eval_episodes"], seed + 1000, step, episode)
            next_eval += config["eval_every"]

        obs, _ = env.reset()
        agent.reset()
        episode_return = 0.0

    if not recorder.evaluations or recorder.evaluations[-1]["step"] < step:
        recorder.evaluate(agent, eval_env, config["eval_episodes"], seed + 1000, step, episode)
    env.close()
    if eval_env is not env:
        eval_env.close()
    writer.close()
    print(f"Évaluations dans {recorder.results_file}, agents dans {recorder.checkpoint_file} et {recorder.best_file}")


if __name__ == "__main__":
    main()
