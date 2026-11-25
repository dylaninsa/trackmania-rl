"""Runs a policy in the game for several episodes and saves the results to a CSV file.

Requires TrackMania open on the track, windowed, in cockpit view, with the TMRL plugin loaded. Click in the
game window right after launching.

Examples:
    Evaluate the PID heuristic on 5 episodes::

        uv run python scripts/evaluate.py --policy pid --episodes 5

    Evaluate the random agent with a given seed::

        uv run python scripts/evaluate.py --policy random --episodes 10 --seed 0

    Evaluate a trained PPO agent::

        uv run python scripts/evaluate.py --policy ppo --config configs/ppo_trackmania.json --checkpoint checkpoints/ppo_trackmania_seed0.pt --episodes 5

    Evaluate a trained SAC agent::

        uv run python scripts/evaluate.py --policy sac --config configs/sac_trackmania.json --checkpoint checkpoints/sac_trackmania_seed0.pt --episodes 5

    Evaluate the PID heuristic on another track, loaded in the game, with longer episodes::

        uv run python scripts/evaluate.py --policy pid --reward-path data/reward/circuit2.pkl --ep-max-length 4500

Attributes:
    TIME_STEP (float): Duration of a step in seconds, used to compute the finish time.
    AGENTS (dict): Agent class of each trained policy, by name on the command line.
"""

import argparse
import csv
import json
from pathlib import Path

from algos.ppo import PPOAgent
from algos.sac import SACAgent
from baselines.policies import PIDPolicy, RandomPolicy
from envs.trackmania import DEFAULT_REWARD_PATH, make_env

TIME_STEP = 0.05
AGENTS = {"ppo": PPOAgent, "sac": SACAgent}


class TrainedPolicy:
    """Deterministic policy of a trained agent, with the interface of the baselines.

    Args:
        agent (PPOAgent | SACAgent): Trained agent.
    """

    def __init__(self, agent):
        self.agent = agent

    def reset(self):
        """Resets the agent at the start of an episode."""
        self.agent.reset()

    def act(self, obs):
        """Chooses the mean action of the policy.

        Args:
            obs (np.ndarray): Observation of size 83.

        Returns:
            np.ndarray: Action in [-1, 1].
        """
        return self.agent.act(obs, deterministic=True)


def make_policy(args, env):
    """Creates the policy to evaluate.

    Args:
        args (argparse.Namespace): Command line arguments: ``policy``, ``seed``, and for a trained agent
            ``config`` and ``checkpoint``.
        env (gymnasium.Env): Environment, whose spaces give the dimensions of the networks.

    Returns:
        RandomPolicy | PIDPolicy | TrainedPolicy: The policy.
    """
    if args.policy == "random":
        return RandomPolicy(args.seed)
    if args.policy in AGENTS:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        agent = AGENTS[args.policy](env.observation_space.shape[0], env.action_space.shape[0], config)
        agent.load(args.checkpoint)
        return TrainedPolicy(agent)
    return PIDPolicy()


def run_episode(env, policy):
    """Plays one full episode with the policy.

    Args:
        env (gymnasium.Env): Environment created by ``make_env``.
        policy (RandomPolicy | PIDPolicy | TrainedPolicy): Policy exposing ``reset()`` and ``act(obs)``.

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


def default_output(args):
    """Chooses the CSV file of the results when ``--output`` is not given.

    The file is ``results/<policy>.csv``, ``results/random_seed<seed>.csv`` for the random agent with a seed,
    and ``results/eval_<checkpoint>.csv`` for a trained agent. On a track other than ``tmrl-test``, the name of
    its trajectory file is appended, so that the results of the two tracks do not overwrite each other.

    Args:
        args (argparse.Namespace): Command line arguments: ``policy``, ``seed``, ``checkpoint`` and
            ``reward_path``.

    Returns:
        pathlib.Path: Output CSV file.
    """
    if args.policy in AGENTS:
        name = f"eval_{Path(args.checkpoint).stem}"
    elif args.policy == "random" and args.seed is not None:
        name = f"random_seed{args.seed}"
    else:
        name = args.policy
    if Path(args.reward_path).resolve() != DEFAULT_REWARD_PATH:
        name += f"_{Path(args.reward_path).stem}"
    return Path("results") / f"{name}.csv"


def main():
    """Evaluates the policy given on the command line and writes one CSV row per episode.

    The output file is ``--output``, or else the one chosen by ``default_output``.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", choices=["random", "pid", *AGENTS], required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None, help="graine de l'agent aléatoire (ignorée par le PID)")
    parser.add_argument("--ep-max-length", type=int, default=3000)
    parser.add_argument("--config", default=None, help="configuration d'entraînement de l'agent")
    parser.add_argument("--checkpoint", default=None, help="fichier .pt de l'agent")
    parser.add_argument("--output", default=None, help="fichier CSV de sortie")
    parser.add_argument("--reward-path", default=DEFAULT_REWARD_PATH,
                        help="trajectoire (.pkl) de la piste chargée dans le jeu")
    args = parser.parse_args()
    if args.policy in AGENTS and (args.config is None or args.checkpoint is None):
        parser.error(f"--policy {args.policy} demande --config et --checkpoint")

    output_file = Path(args.output) if args.output is not None else default_output(args)

    env = make_env(reward_path=args.reward_path, ep_max_length=args.ep_max_length)
    policy = make_policy(args, env)

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
