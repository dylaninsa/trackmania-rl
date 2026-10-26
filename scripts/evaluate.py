"""Runs a policy in the game for several episodes and saves the results to a CSV file.

Requires TrackMania open on the track, windowed, in cockpit view, with the TMRL plugin loaded. Click in the
game window right after launching.

Examples:
    Evaluate the PID heuristic on 5 episodes::

        uv run python scripts/evaluate.py --policy pid --episodes 5

    Evaluate the random agent with a given seed::

        uv run python scripts/evaluate.py --policy random --episodes 10 --seed 0

Attributes:
    TIME_STEP (float): Duration of a step in seconds, used to compute the finish time.
"""

import argparse
import csv
from pathlib import Path

from baselines.policies import PIDPolicy, RandomPolicy
from envs.trackmania import make_env

TIME_STEP = 0.05


def make_policy(name, seed):
    """Creates the policy to evaluate.

    Args:
        name (str): ``"random"`` or ``"pid"``.
        seed (int | None): Seed of the random agent. The PID heuristic is deterministic and ignores it.

    Returns:
        RandomPolicy | PIDPolicy: The policy.
    """
    if name == "random":
        return RandomPolicy(seed)
    return PIDPolicy()


def run_episode(env, policy):
    """Plays one full episode with the policy.

    Args:
        env (gymnasium.Env): Environment created by ``make_env``.
        policy (RandomPolicy | PIDPolicy): Policy exposing ``reset()`` and ``act(obs)``.

    Returns:
        dict: ``return`` (sum of the rewards), ``steps``, ``distance_m`` and ``progress`` at the last step,
            ``finished`` (whether the finish line was crossed) and ``finish_time_s`` (time to reach the finish
            line, None if it was not reached).
    """
    obs, info = env.reset()
    policy.reset()
    episode_return = 0.0
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
        steps += 1
    return {
        "return": episode_return,
        "steps": steps,
        "distance_m": info["distance_m"],
        "progress": info["progress"],
        "finished": info["end_of_track"],
        "finish_time_s": steps * TIME_STEP if info["end_of_track"] else None,
    }


def main():
    """Evaluates the policy given on the command line and writes one CSV row per episode.

    The default output file is ``results/<policy>.csv``, or ``results/random_seed<seed>.csv`` for the random
    agent with a seed, so that the results of different seeds are not overwritten.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", choices=["random", "pid"], required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None, help="graine de l'agent aléatoire (ignorée par le PID)")
    parser.add_argument("--ep-max-length", type=int, default=3000)
    parser.add_argument("--output", default=None, help="fichier CSV de sortie")
    args = parser.parse_args()

    if args.output is not None:
        output_file = Path(args.output)
    elif args.policy == "random" and args.seed is not None:
        output_file = Path("results") / f"random_seed{args.seed}.csv"
    else:
        output_file = Path("results") / f"{args.policy}.csv"

    env = make_env(ep_max_length=args.ep_max_length)
    policy = make_policy(args.policy, args.seed)

    results = []
    for episode in range(1, args.episodes + 1):
        result = {"policy": args.policy, "seed": args.seed, "episode": episode, **run_episode(env, policy)}
        print(f"Épisode {episode} : {result['steps']} pas, retour {result['return']:.2f}, "
              f"distance {result['distance_m']:.1f} m, arrivée : {'oui' if result['finished'] else 'non'}")
        results.append(result)
    env.close()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    finish_times = [r["finish_time_s"] for r in results if r["finished"]]
    mean_distance = sum(r["distance_m"] for r in results) / len(results)
    print(f"\nDistance moyenne : {mean_distance:.1f} m, parcours terminés : {len(finish_times)}/{len(results)}"
          + (f", meilleur temps : {min(finish_times):.2f} s" if finish_times else ""))
    print(f"Résultats enregistrés dans {output_file}")


if __name__ == "__main__":
    main()
