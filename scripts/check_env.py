"""Drives the car with a constant action and prints rewards, progress and step durations.

Requires TrackMania open on the track, windowed, in cockpit view, with the TMRL plugin loaded. Click in the
game window right after launching.

Examples:
    Full throttle straight ahead::

        uv run python scripts/check_env.py

    Check truncation::

        uv run python scripts/check_env.py --ep-max-length 100

    Drive manually with the keyboard::

        uv run python scripts/check_env.py --manual
"""

import argparse
import time

import numpy as np

from envs.trackmania import make_env


def main():
    """Runs the episodes requested on the command line.

    The first and last steps of each episode are left out of the step duration: the first starts during the
    reset and the last includes the pause the environment makes at the end of an episode.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--ep-max-length", type=int, default=3000)
    parser.add_argument("--gas", type=float, default=1.0)
    parser.add_argument("--steer", type=float, default=0.0)
    parser.add_argument("--manual", action="store_true", help="n'envoie aucune action : on conduit au clavier")
    args = parser.parse_args()

    env = make_env(ep_max_length=args.ep_max_length, flatten=not args.manual)
    action = None if args.manual else np.array([args.gas, 0.0, args.steer])

    for episode in range(1, args.episodes + 1):
        env.reset()
        terminated = truncated = False
        step, ep_return, step_ends = 0, 0.0, [time.perf_counter()]
        while not (terminated or truncated):
            _, rew, terminated, truncated, info = env.step(action)
            step_ends.append(time.perf_counter())
            step += 1
            ep_return += rew
            if step % 10 == 0 or terminated or truncated:
                print(f"  pas {step:4d}  r={rew:+.3f}  progression={info['progress']:.3f}  "
                      f"distance={info['distance_m']:7.1f} m  écart={info['dist_to_traj']:5.1f}  "
                      f"vitesse={info['speed']:5.1f}")

        if info["end_of_track"]:
            cause = "arrivée"
        elif terminated:
            cause = "échec"
        else:
            cause = "troncation"
        step_ms = np.diff(step_ends)[1:-1] * 1000
        print(f"Épisode {episode} : {step} pas, retour {ep_return:.2f}, fin = {cause}, "
              f"durée d'un pas {step_ms.mean():.1f} ms (max {step_ms.max():.1f} ms)")

    env.close()


if __name__ == "__main__":
    main()
