import csv
import json
import runpy
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from training.common import EvaluationRecorder, evaluate, make_envs, save_evaluations, score

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


def run_script(script, config, tmp_path, monkeypatch, extra_args=()):
    """Runs a training script on a small configuration, from a temporary working directory.

    Args:
        script (str): File name of the script in ``scripts/``.
        config (dict): Configuration written to ``tmp_path``.
        tmp_path (pathlib.Path): Working directory of the run.
        monkeypatch (pytest.MonkeyPatch): Fixture used to change the directory and the command line.
        extra_args (tuple[str, ...]): Additional command line arguments.

    Returns:
        list[dict]: Rows of the evaluations CSV file.
    """
    config_file = tmp_path / "tiny.json"
    config_file.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [script, "--config", str(config_file), "--seed", "3", *extra_args])
    runpy.run_path(str(SCRIPTS / script), run_name="__main__")
    assert (tmp_path / "checkpoints" / "tiny_seed3.pt").exists()
    assert (tmp_path / "checkpoints" / "tiny_seed3_best.pt").exists()
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

    result = evaluate(agent, env, episodes=3, seed=7)

    assert agent.resets == 3
    assert len(agent.deterministic_calls) == 3 * 200
    assert all(agent.deterministic_calls)
    assert sorted(result) == ["mean_return", "std_return"]
    assert result["mean_return"] < 0.0 and result["std_return"] >= 0.0


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


class FakeTrackEnv:
    """Environment whose episodes last one step and whose ``info`` looks like TrackMania's.

    Args:
        outcomes (list[tuple[float, bool]]): ``(distance_m, end_of_track)`` at the end of each episode.
    """

    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,))

    def reset(self, seed=None):
        """Starts an episode.

        Args:
            seed (int | None): Ignored, like in the game.

        Returns:
            tuple[np.ndarray, dict]: Observation and ``info``.
        """
        return np.zeros(83, dtype=np.float32), {"distance_m": 0.0, "end_of_track": False}

    def step(self, action):
        """Ends the episode with the next outcome.

        Args:
            action (np.ndarray): Ignored.

        Returns:
            tuple: Observation, reward, terminated, truncated and ``info``.
        """
        distance, finished = next(self.outcomes)
        info = {"distance_m": distance, "end_of_track": finished}
        return np.zeros(83, dtype=np.float32), distance / 10, True, False, info


def test_evaluate_reports_distance_and_finish_rate_in_trackmania():
    env = FakeTrackEnv([(100.0, False), (2059.0, True)])

    result = evaluate(ConstantAgent(np.zeros(3, dtype=np.float32)), env, episodes=2, seed=0)

    assert result["mean_distance_m"] == pytest.approx(1079.5)
    assert result["finish_rate"] == pytest.approx(0.5)
    assert result["mean_return"] == pytest.approx(107.95)


def test_save_evaluations_uses_the_keys_as_columns(tmp_path):
    path = tmp_path / "run.csv"

    save_evaluations([{"step": 1, "mean_return": 2.0, "mean_distance_m": 30.0, "finish_rate": 0.0}], path)

    with open(path, encoding="utf-8") as f:
        assert next(csv.reader(f)) == ["step", "mean_return", "mean_distance_m", "finish_rate"]


@pytest.mark.parametrize("script, config", [("train_ppo.py", PPO_CONFIG), ("train_sac.py", SAC_CONFIG)])
def test_max_steps_stops_before_total_steps(script, config, tmp_path, monkeypatch):
    rows = run_script(script, config, tmp_path, monkeypatch, extra_args=("--max-steps", "200"))

    assert [int(row["step"]) for row in rows] == [200]


@pytest.mark.parametrize("script, config", [("train_ppo.py", PPO_CONFIG), ("train_sac.py", SAC_CONFIG)])
def test_final_evaluation_when_the_run_stops_inside_an_episode(script, config, tmp_path, monkeypatch):
    rows = run_script(script, config, tmp_path, monkeypatch, extra_args=("--max-steps", "300"))

    assert [int(row["step"]) for row in rows] == [200, 300]


class SavingAgent(ConstantAgent):
    """Constant agent whose ``save`` records which evaluation each file comes from."""

    def __init__(self):
        super().__init__(np.zeros(3, dtype=np.float32))
        self.saves = []

    def save(self, path):
        """Records the save instead of writing a file.

        Args:
            path (pathlib.Path): Destination file.
        """
        self.saves.append((path.name, self.resets))


class FakeWriter:
    """TensorBoard writer that keeps the scalars in a list."""

    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        """Records a scalar.

        Args:
            tag (str): Name of the scalar.
            value (float): Value.
            step (int): Step.
        """
        self.scalars.append((tag, value, step))


def test_score_uses_distance_in_trackmania_and_return_otherwise():
    assert score({"mean_return": -150.0, "std_return": 3.0}) == -150.0
    assert score({"mean_return": 80.0, "std_return": 0.0, "mean_distance_m": 1287.0, "finish_rate": 0.0}) == 1287.0


def test_recorder_keeps_the_first_best_evaluation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env = FakeTrackEnv([(100.0, False), (2059.0, True), (2059.0, True), (40.0, False)])
    agent = SavingAgent()
    writer = FakeWriter()
    recorder = EvaluationRecorder("run_seed0", writer)

    for step in [10, 20, 30, 40]:
        recorder.evaluate(agent, env, episodes=1, seed=0, step=step, episode=step // 10)

    assert [name for name, _ in agent.saves if name == "run_seed0.pt"] == ["run_seed0.pt"] * 4
    assert [evaluation for name, evaluation in agent.saves if name == "run_seed0_best.pt"] == [1, 2]
    assert recorder.best_score == 2059.0
    assert ("eval/mean_distance_m", 40.0, 40) in writer.scalars
    with open(tmp_path / "results" / "run_seed0.csv", encoding="utf-8") as f:
        assert [row["step"] for row in csv.DictReader(f)] == ["10", "20", "30", "40"]
    assert (tmp_path / "checkpoints").is_dir()
