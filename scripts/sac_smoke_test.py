"""Smoke test of the chain environment -> replay buffer -> SAC agent, on the real game.

This is an integration test, not a training script: the agent is too little trained to learn anything. It
checks that the pieces written by different members fit together on real data, and it measures the timings
needed to size the training loop. The run is split in four phases:

1. warm-up: random actions fill the replay buffer;
2. a few short episodes with the stochastic SAC policy, with gradient updates between the episodes only
   (the game does not pause during an episode);
3. one deterministic evaluation episode;
4. a checkpoint round trip (``save`` then ``load`` gives the same policy).

Contract violations are collected and listed at the end instead of stopping the run, and the exit code is 1
if there is any. TrackMania must be open on the tmrl-test track, like for ``check_env.py``.

Example::

    uv run python scripts/sac_smoke_test.py
    uv run python scripts/sac_smoke_test.py --device cuda --episodes 5
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algos.replay_buffer import ReplayBuffer  # noqa: E402
from algos.sac import SACAgent  # noqa: E402
from envs.trackmania import ACTIONS, LIDAR, make_env  # noqa: E402

STEP_MS = 50.0
INFO_KEYS = ("speed", "end_of_track", "progress", "distance_m", "dist_to_traj")


class Checks:
    """Collects the contract violations found during the run instead of stopping at the first one."""

    def __init__(self):
        self.problems = []

    def ok(self, condition, message):
        """Records ``message`` as a problem (once) when ``condition`` is false.

        Args:
            condition (bool): What should hold.
            message (str): Description of the violation.

        Returns:
            bool: ``condition``.
        """
        if not condition and message not in self.problems:
            self.problems.append(message)
            print(f"  !! {message}")
        return bool(condition)


def check_reset(obs, info, obs_dim, checks):
    """Checks the reset contract of the README: shape, identical scans, zero actions, ``info`` keys."""
    if not checks.ok(obs.shape == (obs_dim,) and obs.dtype == np.float32,
                     f"reset : observation de forme {obs.shape} et de type {obs.dtype}"):
        return
    scans = obs[LIDAR].reshape(4, -1)
    checks.ok(np.allclose(scans, scans[0]), "reset : les 4 balayages LIDAR devraient être identiques")
    checks.ok(np.allclose(obs[ACTIONS], 0.0), "reset : les 2 dernières actions devraient valoir 0")
    checks.ok(all(k in info for k in INFO_KEYS), f"reset : clés manquantes dans info (attendues : {INFO_KEYS})")


def run_episode(env, agent, buffer, obs_dim, act_dim, checks, deterministic=False, store=True):
    """Runs one episode without any gradient update.

    Args:
        env (gymnasium.Env): TrackMania environment.
        agent (SACAgent | None): Policy, or None for uniformly random actions.
        buffer (ReplayBuffer): Receives the transitions if ``store`` is True. Only ``terminated`` is stored:
            a truncated episode could have continued.
        obs_dim (int): Expected size of an observation.
        act_dim (int): Expected size of an action.
        checks (Checks): Collects contract violations.
        deterministic (bool): Passed to ``agent.act``.
        store (bool): Whether to add the transitions to the buffer.

    Returns:
        dict: Return, number of steps, end cause, distance, progress, first observation, and the
            durations of ``act`` and of a whole step in milliseconds.
    """
    obs, info = env.reset()
    check_reset(obs, info, obs_dim, checks)
    first_obs = obs
    ret, steps, act_ms = 0.0, 0, []
    step_ends = [time.perf_counter()]
    terminated = truncated = False

    while not (terminated or truncated):
        t0 = time.perf_counter()
        act = env.action_space.sample() if agent is None else agent.act(obs, deterministic=deterministic)
        act_ms.append(1000 * (time.perf_counter() - t0))
        act = np.asarray(act, dtype=np.float32)
        checks.ok(act.shape == (act_dim,) and np.all(np.isfinite(act)) and np.all(np.abs(act) <= 1.0),
                  f"action invalide : {act}")

        next_obs, rew, terminated, truncated, info = env.step(act)
        step_ends.append(time.perf_counter())
        checks.ok(next_obs.shape == (obs_dim,) and np.all(np.isfinite(next_obs)), "observation invalide au step")
        checks.ok(rew >= 0.0, f"récompense négative ({rew}) alors que le README dit qu'elle ne l'est jamais")

        if store:
            buffer.add(obs, act, rew, next_obs, terminated)
        obs = next_obs
        ret += float(rew)
        steps += 1

    # like check_env: the first step starts during the reset, the last one includes the end-of-episode pause
    step_ms = np.diff(step_ends)[1:-1] * 1000
    cause = "arrivée" if info["end_of_track"] else ("échec" if terminated else "troncation")
    return {"return": ret, "steps": steps, "cause": cause, "distance_m": info["distance_m"],
            "progress": info["progress"], "first_obs": first_obs, "act_ms": np.array(act_ms),
            "step_ms": step_ms}


def describe(s):
    """Formats the statistics of an episode on one line."""
    step = f"{s['step_ms'].mean():.1f} ms (max {s['step_ms'].max():.1f})" if s["step_ms"].size else "n/a"
    return (f"{s['steps']} pas, retour {s['return']:.2f}, distance {s['distance_m']:.1f} m, "
            f"fin = {s['cause']} | act() {s['act_ms'].mean():.2f} ms (max {s['act_ms'].max():.2f}) | "
            f"pas {step}")


def check_checkpoint(agent, config, obs_dim, act_dim, obs, checks):
    """Saves the agent, loads it in a new one and checks that both give the same deterministic action."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "smoke" / "agent.pt"
        agent.save(path)
        clone = SACAgent(obs_dim, act_dim, config)
        clone.load(path)
    same = np.allclose(agent.act(obs, deterministic=True), clone.act(obs, deterministic=True), atol=1e-6)
    checks.ok(same, "checkpoint : l'agent rechargé ne donne pas la même action déterministe")
    print("  checkpoint save/load : " + ("identique" if same else "DIFFÉRENT"))


def main():
    """Runs the four phases and prints a summary; exits with 1 if a contract violation was found."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--warmup-steps", type=int, default=400, help="random steps before the first update")
    parser.add_argument("--episodes", type=int, default=3, help="SAC episodes after the warm-up")
    parser.add_argument("--max-steps", type=int, default=300, help="ep_max_length: steps before truncation")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates-per-step", type=float, default=0.5, help="update-to-data ratio")
    parser.add_argument("--max-updates", type=int, default=500, help="cap on the updates between two episodes")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checks = Checks()
    env = make_env(ep_max_length=args.max_steps)
    env.action_space.seed(args.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"obs_dim={obs_dim} act_dim={act_dim} device={args.device}")
    checks.ok(obs_dim == 83 and act_dim == 3, f"contrat de l'environnement : attendu 83 / 3, obtenu {obs_dim} / {act_dim}")

    config = {"hidden": 256, "lr": 3e-4, "gamma": 0.99, "tau": 0.005, "device": args.device}
    agent = SACAgent(obs_dim, act_dim, config)
    buffer = ReplayBuffer(obs_dim, act_dim, capacity=100_000, device=args.device, seed=args.seed)
    for _ in range(3):  # the first forward passes are slow (lazy initialization); keep them out of the timings
        agent.act(np.zeros(obs_dim, dtype=np.float32))

    sac_step_ms, update_ms, last_obs = [], [], None
    try:
        print(f"\n[1] Warm-up : {args.warmup_steps} pas aléatoires")
        collected = 0
        while collected < args.warmup_steps:
            s = run_episode(env, None, buffer, obs_dim, act_dim, checks)
            collected += s["steps"]
            last_obs = s["first_obs"]
            print(f"  {describe(s)}  (buffer = {len(buffer)})")

        for ep in range(1, args.episodes + 1):
            print(f"\n[2] Épisode SAC {ep}/{args.episodes} (politique stochastique)")
            s = run_episode(env, agent, buffer, obs_dim, act_dim, checks)
            sac_step_ms.append(s["step_ms"])
            print(f"  {describe(s)}")
            checks.ok(s["steps"] <= args.max_steps, "épisode plus long que ep_max_length")

            n_up = min(int(s["steps"] * args.updates_per_step), args.max_updates)
            if len(buffer) < args.batch_size or n_up == 0:
                print("  pas de mise à jour (buffer trop petit)")
                continue
            t0 = time.perf_counter()
            for _ in range(n_up):
                metrics = agent.update(buffer.sample(args.batch_size))
            dt = time.perf_counter() - t0
            update_ms.append(1000 * dt / n_up)
            checks.ok(all(np.isfinite(v) for v in metrics.values()), f"métriques non finies : {metrics}")
            checks.ok(abs(metrics["critic_loss"]) < 1e3, f"critic_loss anormalement grande : {metrics['critic_loss']:.1f}")
            print(f"  {n_up} mises à jour en {dt:.1f} s ({update_ms[-1]:.1f} ms chacune) | "
                  f"critic={metrics['critic_loss']:.3f} actor={metrics['actor_loss']:.3f} "
                  f"alpha={metrics['alpha']:.3f} q_mean={metrics['q_mean']:.3f} entropy={metrics['entropy']:.2f}")

        print("\n[3] Évaluation (politique déterministe)")
        s = run_episode(env, agent, buffer, obs_dim, act_dim, checks, deterministic=True, store=False)
        sac_step_ms.append(s["step_ms"])
        print(f"  {describe(s)}")

        print("\n[4] Checkpoint")
        check_checkpoint(agent, config, obs_dim, act_dim, last_obs, checks)
    finally:
        env.close()

    print("\n=== Résumé ===")
    steps = np.concatenate(sac_step_ms) if sac_step_ms else np.array([])
    if steps.size:
        print(f"durée d'un pas avec SAC : {steps.mean():.1f} ms en moyenne, max {steps.max():.1f} ms "
              f"(budget {STEP_MS:.0f} ms)")
        checks.ok(steps.mean() <= 1.1 * STEP_MS, "les pas durent plus que le budget de 50 ms : le calcul est trop lent")
    if update_ms:
        per_episode = np.mean(update_ms) * 3000 / 1000
        print(f"mise à jour : {np.mean(update_ms):.1f} ms en moyenne -> {per_episode:.0f} s de calcul pour 3000 "
              f"mises à jour (un épisode complet de 150 s au ratio 1:1)")
    if checks.problems:
        print(f"\n{len(checks.problems)} problème(s) :")
        for p in checks.problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\nAucun problème : la chaîne env -> buffer -> SAC fonctionne (aucun apprentissage n'est attendu ici).")


if __name__ == "__main__":
    main()