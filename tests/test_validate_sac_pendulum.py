import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "validate_sac_pendulum.py"
spec = importlib.util.spec_from_file_location("validate_sac_pendulum", SCRIPT)
validate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate)

TINY_CONFIG = {"env": "Pendulum-v1", "seed": 0, "total_steps": 300, "start_steps": 100, "buffer_size": 1000,
               "batch_size": 32, "eval_every": 150, "eval_episodes": 1, "hidden": 16, "lr": 3e-4, "gamma": 0.99,
               "tau": 0.005, "device": "cpu"}


def make_rows(returns, step_size=2000):
    """Builds evaluation rows with the given returns, one every ``step_size`` steps."""
    return [{"step": i * step_size, "eval_return": r} for i, r in enumerate(returns)]


def test_train_returns_one_row_per_evaluation():
    rows, non_finite = validate.train(TINY_CONFIG, seed=0, log=lambda _: None)

    assert [r["step"] for r in rows] == [0, 150, 300]
    assert all(set(validate.COLUMNS) <= set(r) for r in rows)
    assert not non_finite
    assert np.isfinite(rows[-1]["alpha"])


def test_train_is_reproducible_with_a_seed():
    first, _ = validate.train(TINY_CONFIG, seed=3, log=lambda _: None)
    second, _ = validate.train(TINY_CONFIG, seed=3, log=lambda _: None)

    assert [r["eval_return"] for r in first] == [r["eval_return"] for r in second]


def test_write_csv(tmp_path):
    rows, _ = validate.train(TINY_CONFIG, seed=0, log=lambda _: None)
    path = tmp_path / "out" / "run.csv"

    validate.write_csv(rows, path)

    assert path.read_text(encoding="utf-8").splitlines()[0] == ",".join(validate.COLUMNS)
    assert len(path.read_text(encoding="utf-8").splitlines()) == len(rows) + 1


def test_check_run_accepts_a_good_run():
    assert validate.check_run(make_rows([-1200, -800, -300, -150, -140, -130]), non_finite=False) == []


def test_check_run_rejects_non_finite_metrics():
    problems = validate.check_run(make_rows([-1200, -300, -150, -140]), non_finite=True)

    assert len(problems) == 1 and "finies" in problems[0]


def test_check_run_rejects_a_run_that_never_learns():
    problems = validate.check_run(make_rows([-1200, -1100, -1000, -1000, -900, -950]), non_finite=False)

    assert len(problems) == 2


def test_check_run_rejects_a_collapse():
    problems = validate.check_run(make_rows([-1200, -150, -140, -130, -900, -1000]), non_finite=False)

    assert len(problems) == 1 and "finale" in problems[0]