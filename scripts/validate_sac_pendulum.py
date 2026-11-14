"""Validates the SAC implementation on Pendulum-v1, without the game.

Pendulum-v1 is a light environment where a correct SAC learns in a few thousand steps. If it does not, the
bug is in the algorithm and not in TrackMania. The run follows the hyperparameters of
``configs/sac_pendulum.json`` and does one gradient update per environment step after the warm-up.

Every ``eval_every`` steps, the deterministic policy is evaluated on ``eval_episodes`` episodes whose seeds
are separate from the training seed. One CSV per seed is written to ``results/``. At the end, the success
criteria are checked and the exit code is 1 if one of them fails:

1. all the metrics of ``update`` stay finite;
2. an evaluation return above ``SUCCESS_RETURN`` is reached within ``SUCCESS_STEPS`` steps;
3. the policy does not collapse: the mean of the last two evaluations is above ``SUCCESS_RETURN``.

Examples:
    Run the seed of the config::

        uv run python scripts/validate_sac_pendulum.py

    Run three seeds (the protocol of the report)::

        uv run python scripts/validate_sac_pendulum.py --seeds 0 1 2

Attributes:
    SUCCESS_RETURN (float): Evaluation return above which Pendulum-v1 is considered solved.
    SUCCESS_STEPS (int): Number of steps within which ``SUCCESS_RETURN`` must be reached.
    RANDOM_RETURN (float): Approximate return of a random policy, for reference.
    COLUMNS (tuple[str, ...]): Columns of the CSV files.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algos.replay_buffer import ReplayBuffer  # noqa: E402
from algos.sac import SACAgent  # noqa: E402

SUCCESS_RETURN = -250.0
SUCCESS_STEPS = 10_000
RANDOM_RETURN = -1200.0
EVAL_SEED_OFFSET = 10_000
COLUMNS = ("step", "eval_return", "eval_std", "train_return", "critic_loss", "actor_loss", "alpha", "q_mean",
           "entropy", "elapsed_s")
METRIC_KEYS = ("critic_loss", "actor_loss", "alpha", "q_mean", "entropy")


def evaluate(agent, env, episodes, seed):
    """Plays episodes with the deterministic policy.

    Args:
        agent (SACAgent): Agent to evaluate.
        env (gymnasium.Env): Evaluation environment, separate from the training one.
        episodes (int): Number of episodes.
        seed (int): Seed of the first episode; the others use consecutive seeds.

    Returns:
        tuple[float, float]: Mean and standard deviation of the episode returns.
    """
    scale = float(env.action_space.high[0])
    returns = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        total, done = 0.0, False
        while not done:
            action = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action * scale)
            total += float(reward)
            done = terminated or truncated
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))


def train(config, seed, log=print):
    """Trains SAC on the environment of the config and evaluates it regularly.

    The agent outputs actions in [-1, 1], multiplied by the action bound of the environment (2 for
    Pendulum-v1) before being sent. Only ``terminated`` is stored in the buffer: Pendulum-v1 never terminates,
    it is truncated after 200 steps, and a truncated transition must still be bootstrapped.

    Args:
        config (dict): Content of ``configs/sac_pendulum.json``.
        seed (int): Seed of the networks, the buffer, the exploration and the environment.
        log (Callable[[str], None]): Function that receives the progress messages.

    Returns:
        tuple[list[dict], bool]: One row per evaluation (keys of ``COLUMNS``, with the metrics averaged over
            the updates since the previous evaluation), and whether a metric was not finite.
    """
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    env = gym.make(config["env"])
    eval_env = gym.make(config["env"])
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    scale = float(env.action_space.high[0])

    agent = SACAgent(obs_dim, act_dim, config)
    buffer = ReplayBuffer(obs_dim, act_dim, capacity=config["buffer_size"], device=config["device"], seed=seed)

    rows, non_finite = [], False
    window = {k: [] for k in METRIC_KEYS}
    episode_returns, episode_return = [], 0.0
    start = time.perf_counter()

    def record(step):
        mean, std = evaluate(agent, eval_env, config["eval_episodes"], EVAL_SEED_OFFSET + 1000 * seed)
        row = {"step": step, "eval_return": mean, "eval_std": std,
               "train_return": float(np.mean(episode_returns)) if episode_returns else float("nan"),
               "elapsed_s": time.perf_counter() - start}
        for k in METRIC_KEYS:
            row[k] = float(np.mean(window[k])) if window[k] else float("nan")
            window[k].clear()
        episode_returns.clear()
        rows.append(row)
        log(f"  graine {seed} | pas {step:6d} | éval {mean:8.1f} (± {std:5.1f}) | train {row['train_return']:8.1f} | "
            f"alpha {row['alpha']:.3f} | q {row['q_mean']:8.2f} | entropie {row['entropy']:6.2f} | "
            f"{row['elapsed_s']:5.0f} s")

    record(0)  # untrained agent: should be close to the random return
    obs, _ = env.reset(seed=seed)
    for step in range(1, config["total_steps"] + 1):
        if step <= config["start_steps"]:
            action = rng.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
        else:
            action = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action * scale)
        buffer.add(obs, action, reward, next_obs, terminated)
        episode_return += float(reward)
        obs = next_obs
        if terminated or truncated:
            episode_returns.append(episode_return)
            episode_return = 0.0
            obs, _ = env.reset()

        if step > config["start_steps"] and len(buffer) >= config["batch_size"]:
            metrics = agent.update(buffer.sample(config["batch_size"]))
            if not all(np.isfinite(v) for v in metrics.values()):
                non_finite = True
            for k in METRIC_KEYS:
                window[k].append(metrics[k])

        if step % config["eval_every"] == 0:
            record(step)

    env.close()
    eval_env.close()
    return rows, non_finite


def check_run(rows, non_finite):
    """Checks the success criteria of one run.

    Args:
        rows (list[dict]): Rows returned by ``train``.
        non_finite (bool): Whether a metric of ``update`` was not finite.

    Returns:
        list[str]: Description of each failed criterion, empty if the run is valid.
    """
    problems = []
    if non_finite:
        problems.append("des métriques de update ne sont pas finies")
    early = [r["eval_return"] for r in rows if r["step"] <= SUCCESS_STEPS]
    if max(early) <= SUCCESS_RETURN:
        problems.append(f"retour d'évaluation jamais supérieur à {SUCCESS_RETURN:.0f} avant {SUCCESS_STEPS} pas "
                        f"(meilleur : {max(early):.1f})")
    last = float(np.mean([r["eval_return"] for r in rows[-2:]]))
    if last <= SUCCESS_RETURN:
        problems.append(f"politique finale insuffisante : moyenne des deux dernières évaluations {last:.1f} "
                        f"<= {SUCCESS_RETURN:.0f}")
    return problems


def write_csv(rows, path):
    """Writes the rows of a run to a CSV file.

    Args:
        rows (list[dict]): Rows returned by ``train``.
        path (pathlib.Path): Destination file; its folder is created if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Runs the seeds given on the command line and exits with 1 if a success criterion fails."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/sac_pendulum.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="graines à lancer (défaut : celle du config)")
    parser.add_argument("--total-steps", type=int, default=None, help="remplace total_steps du config")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    if args.total_steps is not None:
        config["total_steps"] = args.total_steps
    seeds = args.seeds if args.seeds is not None else [config["seed"]]

    print(f"{config['env']} | {config['total_steps']} pas | graines {seeds} | device {config['device']}")
    print(f"référence : politique aléatoire ≈ {RANDOM_RETURN:.0f}, critère de réussite > {SUCCESS_RETURN:.0f}")

    failed = {}
    for seed in seeds:
        print(f"\n[graine {seed}]")
        rows, non_finite = train(config, seed)
        path = Path(args.output_dir) / f"sac_pendulum_seed{seed}.csv"
        write_csv(rows, path)
        print(f"  résultats enregistrés dans {path}")
        problems = check_run(rows, non_finite)
        if problems:
            failed[seed] = problems

    print("\n=== Résumé ===")
    if failed:
        for seed, problems in failed.items():
            print(f"graine {seed} : ÉCHEC")
            for p in problems:
                print(f"  - {p}")
        sys.exit(1)
    print(f"Les {len(seeds)} graine(s) respectent les critères : SAC apprend sur {config['env']}.")


if __name__ == "__main__":
    main()