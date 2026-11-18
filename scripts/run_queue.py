"""Runs a queue of trainings one after the other, for long unattended sessions.

The queue is a JSON list of runs ``{"algo": "ppo" | "sac", "config": ..., "seed": ..., "max_steps": ...}``,
``max_steps`` being optional. Each run is launched in its own process and its output goes to
``logs/<config>_seed<seed>.log``. When a run ends without error, its evaluations, its log and its TensorBoard
logs are copied to ``experiments/<config>_seed<seed>/`` (``eval.csv``, ``train.log``, ``tensorboard/``), to be
committed, and it leaves a ``logs/<config>_seed<seed>.done`` marker. The runs that have one are skipped, so that the queue can be restarted after an interruption
without redoing the finished runs. A run that crashes is reported and the queue goes on.

Example::

    uv run python scripts/run_queue.py experiments/queue.json

Attributes:
    SCRIPTS (dict): Training script of each algorithm.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPTS = {"ppo": "scripts/train_ppo.py", "sac": "scripts/train_sac.py"}


def run_name(run):
    """Builds the name of a run, the same as its results and checkpoint files.

    Args:
        run (dict): Run of the queue, with ``config`` and ``seed``.

    Returns:
        str: ``<config>_seed<seed>``.
    """
    return f"{Path(run['config']).stem}_seed{run['seed']}"


def command(run):
    """Builds the command line that trains a run.

    Args:
        run (dict): Run of the queue.

    Returns:
        list[str]: Arguments of the training process.
    """
    args = [sys.executable, SCRIPTS[run["algo"]], "--config", run["config"], "--seed", str(run["seed"])]
    if "max_steps" in run:
        args += ["--max-steps", str(run["max_steps"])]
    return args


def archive(name, logs):
    """Copies the files of a finished run to ``experiments/<name>/``, so that they can be committed.

    Args:
        name (str): Name of the run.
        logs (pathlib.Path): Folder of the logs.
    """
    folder = Path("experiments") / name
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    shutil.copy(Path("results") / f"{name}.csv", folder / "eval.csv")
    shutil.copy(logs / f"{name}.log", folder / "train.log")
    shutil.copytree(Path("runs") / name, folder / "tensorboard")


def main():
    """Runs every unfinished run of the queue given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("queue", help="fichier JSON de la file d'attente")
    parser.add_argument("--logs", default="logs", help="dossier des logs et des marqueurs de fin")
    args = parser.parse_args()

    runs = json.loads(Path(args.queue).read_text(encoding="utf-8"))
    logs = Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}

    for index, run in enumerate(runs, start=1):
        name = run_name(run)
        done_file = logs / f"{name}.done"
        if done_file.exists():
            print(f"[{index}/{len(runs)}] {name} : déjà terminé, ignoré")
            continue

        print(f"[{index}/{len(runs)}] {name} : lancé le {datetime.now():%d/%m à %H:%M}", flush=True)
        shutil.rmtree(Path("runs") / name, ignore_errors=True)
        start = time.time()
        with open(logs / f"{name}.log", "w", encoding="utf-8") as log:
            code = subprocess.call(command(run), stdout=log, stderr=subprocess.STDOUT, env=env)
        hours = (time.time() - start) / 3600

        if code == 0:
            archive(name, logs)
            done_file.write_text(f"{datetime.now():%Y-%m-%d %H:%M}, {hours:.2f} h\n", encoding="utf-8")
            print(f"    terminé en {hours:.1f} h, résultats copiés dans experiments/{name}", flush=True)
        else:
            print(f"    ÉCHEC (code {code}) après {hours:.1f} h, voir {logs / name}.log", flush=True)


if __name__ == "__main__":
    main()
