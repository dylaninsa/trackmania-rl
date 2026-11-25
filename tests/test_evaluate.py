import argparse
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "evaluate.py"


@pytest.fixture
def evaluate_script():
    """Loads ``scripts/evaluate.py`` as a module, since ``scripts/`` is not a package.

    Returns:
        module: The ``evaluate`` module.
    """
    spec = importlib.util.spec_from_file_location("evaluate_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_args(evaluate_script, **kwargs):
    """Builds command line arguments with the defaults of ``evaluate.py``.

    Args:
        evaluate_script (module): The ``evaluate`` module.
        **kwargs: Arguments to change.

    Returns:
        argparse.Namespace: The arguments.
    """
    args = {"policy": "pid", "seed": None, "checkpoint": None, "reward_path": evaluate_script.DEFAULT_REWARD_PATH}
    return argparse.Namespace(**{**args, **kwargs})


@pytest.mark.parametrize("kwargs, expected", [
    ({}, "pid.csv"),
    ({"policy": "random"}, "random.csv"),
    ({"policy": "random", "seed": 3}, "random_seed3.csv"),
    ({"policy": "sac", "checkpoint": "checkpoints/sac_trackmania_p3_time_seed4_best.pt"},
     "eval_sac_trackmania_p3_time_seed4_best.csv"),
])
def test_default_output_on_the_training_track(evaluate_script, kwargs, expected):
    assert evaluate_script.default_output(make_args(evaluate_script, **kwargs)) == Path("results") / expected


def test_default_output_recognizes_the_training_track_given_as_a_relative_path(evaluate_script, monkeypatch):
    monkeypatch.chdir(evaluate_script.DEFAULT_REWARD_PATH.parents[2])

    args = make_args(evaluate_script, reward_path="data/reward/tmrl-test.pkl")

    assert evaluate_script.default_output(args) == Path("results") / "pid.csv"


@pytest.mark.parametrize("kwargs, expected", [
    ({}, "pid_circuit2.csv"),
    ({"policy": "random", "seed": 3}, "random_seed3_circuit2.csv"),
    ({"policy": "sac", "checkpoint": "checkpoints/sac_trackmania_p3_base_seed0_best.pt"},
     "eval_sac_trackmania_p3_base_seed0_best_circuit2.csv"),
])
def test_default_output_names_the_other_track(evaluate_script, kwargs, expected):
    args = make_args(evaluate_script, reward_path="data/reward/circuit2.pkl", **kwargs)

    assert evaluate_script.default_output(args) == Path("results") / expected
