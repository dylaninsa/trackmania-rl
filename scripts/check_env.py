"""Test d'intégration de l'environnement avec le jeu.

Prérequis : TrackMania ouvert sur la piste, en fenêtré, caméra cockpit (touche 3), plugin TMRL chargé.
Cliquer dans la fenêtre du jeu juste après le lancement.

    uv run python scripts/check_env.py                       # plein gaz tout droit, 3 épisodes
    uv run python scripts/check_env.py --ep-max-length 100   # vérifier la troncation
    uv run python scripts/check_env.py --manual              # conduire soi-même au clavier
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.trackmania import DEFAULT_REWARD_PATH, make_env  # noqa: E402
from envs.wrappers import LIDAR, OBS_DIM  # noqa: E402

TIME_STEP = 0.05


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--ep-max-length", type=int, default=1000)
    parser.add_argument("--gas", type=float, default=1.0)
    parser.add_argument("--steer", type=float, default=0.0)
    parser.add_argument("--manual", action="store_true", help="n'envoie aucune action : on conduit au clavier")
    parser.add_argument("--reward-path", default=str(DEFAULT_REWARD_PATH))
    parser.add_argument("--print-every", type=int, default=10)
    args = parser.parse_args()

    # En manuel, l'action None se retrouve dans l'observation : on garde le tuple brut de TMRL.
    env = make_env(reward_path=args.reward_path, ep_max_length=args.ep_max_length, flatten=not args.manual)
    action = None if args.manual else np.array([args.gas, 0.0, args.steer], dtype=np.float32)

    obs, info = env.reset()
    if args.manual:
        print("Observation brute :", [np.shape(o) for o in obs])
    else:
        assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32, (obs.shape, obs.dtype)
        assert env.observation_space.contains(obs), "observation hors de observation_space"
        print(f"Observation : {obs.shape} {obs.dtype}")
    print("Info au reset :", info)

    max_speed, max_lidar = 0.0, 0.0
    for episode in range(args.episodes):
        if episode > 0:
            obs, info = env.reset()
            ok = info["traj_index"] == 0 and info["progress"] == 0.0
            print(f"Reset : progression remise à zéro {'OK' if ok else 'ÉCHEC'} ({info['progress']:.3f})")

        ep_return, durations, terminated, truncated, step = 0.0, [], False, False, 0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t_prev = time.perf_counter()
            while not (terminated or truncated):
                obs, rew, terminated, truncated, info = env.step(action)
                now = time.perf_counter()
                durations.append(now - t_prev)
                t_prev = now
                step += 1
                ep_return += float(rew)
                max_speed = max(max_speed, info["speed"])
                lidar = obs[1] if args.manual else obs[LIDAR] * env.lidar_scale
                max_lidar = max(max_lidar, float(np.max(lidar)))
                if step % args.print_every == 0 or terminated or truncated:
                    print(f"  pas {step:4d}  r={float(rew):+.3f}  progression={info['progress']:.3f}  "
                          f"distance={info['distance_m']:7.1f} m  écart={info['dist_to_traj']:5.1f}  "
                          f"vitesse={info['speed']:6.1f}")

        durations = np.array(durations[1:])  # le premier pas inclut le temps de démarrage
        cause = "arrivée" if info["end_of_track"] else "échec (bloqué / hors trajectoire)" if terminated else "troncation"
        print(f"Épisode {episode + 1} : {step} pas, retour {ep_return:.2f}, fin = {cause}")
        if len(durations):
            late = int(np.sum(durations > TIME_STEP * 1.1))
            print(f"  durée d'un pas : moyenne {durations.mean() * 1000:.1f} ms, max {durations.max() * 1000:.1f} ms, "
                  f"{late} pas > {TIME_STEP * 1100:.0f} ms")
        timeouts = [w for w in caught if "time" in str(w.message).lower()]
        if timeouts:
            print(f"  {len(timeouts)} avertissement(s) de dépassement de temps rtgym, ex. : {timeouts[0].message}")

    print(f"\nMaxima bruts observés (pour calibrer la normalisation) : vitesse {max_speed:.1f}, LIDAR {max_lidar:.1f}")
    env.close()


if __name__ == "__main__":
    main()
