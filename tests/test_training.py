import csv
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from training.common import evaluate, make_envs, save_evaluations

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
COMMON = {"env": "Pendulum-v1", "seed": 0, "ep_max_length": 200, "eval_episodes": 1, "device": "cpu"}
PPO_CONFIG = {
    **COMMON,
    "total_steps": 400,
    "buffer_size": 200,
    "batch_size": 64,
    "ppo_epochs": 1,
    "eval_every": 200,
    "hidden": 16,
    "lr_actor": 1e-3,
    "lr_critic": 1e-3,
    "min_lr_fraction": 0.1,
    "gamma": 0.95,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "entropy_coef": 0.0,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "normalize_states": True,
    "reward_scale": 0.1,
}
SAC_CONFIG = {
    **COMMON,
    "total_steps": 400,
    "start_steps": 200,
    "buffer_size": 1000,
    "batch_size": 32,
    "eval_every": 200,
    "hidden": 16,
    "lr": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "updates_per_step": 0.25,
}


class ConstantAgent:
    """Agent that always plays the same action and counts its calls.

    Args:
        action (np.ndarray): Action in [-1, 1].
    """

    def __init__(self, action):
        self.action = action
        self.resets = 0
        self.deterministic_calls = []

    def reset(self):
        """Counts the episodes."""
        self.resets += 1

    def act(self, obs, deterministic=False):
        """Returns the constant action.

        Args:
            obs (np.ndarray): Observation, ignored.
            deterministic (bool): Recorded, to check that the evaluation asks for the mean action.

        Returns:
            np.ndarray: The constant action.
        """
        self.deterministic_calls.append(deterministic)
        return self.action


def run_script(script, config, tmp_path, monkeypatch):
    """Runs a training script on a small configuration, from a temporary working directory.

    Args:
        script (str): File name of the script in ``scripts/``.
        config (dict): Configuration written to ``tmp_path``.
        tmp_path (pathlib.Path): Working directory of the run.
        monkeypatch (pytest.MonkeyPatch): Fixture used to change the directory and the command line.

    Returns:
        list[dict]: Rows of the evaluations CSV file.
    """
    config_file = tmp_path / "tiny.json"
    config_file.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [script, "--config", str(config_file), "--seed", "3"])
    runpy.run_path(str(SCRIPTS / script), run_name="__main__")
    assert (tmp_path / "checkpoints" / "tiny_seed3.pt").exists()
    with open(tmp_path / "results" / "tiny_seed3.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_make_envs_creates_two_separate_gym_envs():
    env, eval_env = make_envs({"env": "Pendulum-v1", "ep_max_length": 200})

    assert env is not eval_env
    assert env.observation_space.shape == (3,)
    env.close()
    eval_env.close()


def test_evaluate_plays_deterministic_episodes():
    env, _ = make_envs({"env": "Pendulum-v1", "ep_max_length": 200})
    agent = ConstantAgent(np.zeros(1, dtype=np.float32))

    mean, std = evaluate(agent, env, episodes=3, seed=7)

    assert agent.resets == 3
    assert len(agent.deterministic_calls) == 3 * 200
    assert all(agent.deterministic_calls)
    assert mean < 0.0 and std >= 0.0


def test_evaluate_is_reproducible_with_the_same_seed():
    env, _ = make_envs({"env": "Pendulum-v1", "ep_max_length": 200})
    agent = ConstantAgent(np.ones(1, dtype=np.float32))

    assert evaluate(agent, env, episodes=2, seed=7) == evaluate(agent, env, episodes=2, seed=7)


def test_evaluate_rescales_the_action_to_the_env_bounds():
    env, _ = make_envs({"env": "Pendulum-v1", "ep_max_length": 200})
    seen = []
    step = env.step
    env.step = lambda action: seen.append(action) or step(action)

    evaluate(ConstantAgent(np.full(1, 0.5, dtype=np.float32)), env, episodes=1, seed=0)

    assert seen[0] == pytest.approx([0.5 * env.action_space.high[0]])


def test_save_evaluations_creates_folder_and_overwrites(tmp_path):
    path = tmp_path / "results" / "run.csv"

    save_evaluations([{"step": 1, "mean_return": -5.0, "std_return": 1.0}], path)
    save_evaluations([{"step": 2, "mean_return": -3.0, "std_return": 0.5}], path)

    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows == [{"step": "2", "mean_return": "-3.0", "std_return": "0.5"}]


def test_train_ppo_runs_end_to_end(tmp_path, monkeypatch):
    rows = run_script("train_ppo.py", PPO_CONFIG, tmp_path, monkeypatch)

    assert [int(row["step"]) for row in rows] == [200, 400]


def test_train_sac_runs_end_to_end(tmp_path, monkeypatch):
    rows = run_script("train_sac.py", SAC_CONFIG, tmp_path, monkeypatch)

    assert [int(row["step"]) for row in rows] == [200, 400]
    assert all(np.isfinite(float(row["mean_return"])) for row in rows)
