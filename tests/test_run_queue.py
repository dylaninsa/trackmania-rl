import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_queue.py"
FAKE_TRAINING = """import json, sys
from pathlib import Path
with open("calls.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(sys.argv[1:]) + "\\n")
print("Pas 200 : é")
sys.exit({code})
"""


@pytest.fixture
def run_queue():
    """Loads ``scripts/run_queue.py`` as a module, since ``scripts/`` is not a package.

    Returns:
        module: The ``run_queue`` module.
    """
    spec = importlib.util.spec_from_file_location("run_queue", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def start_queue(run_queue, runs, tmp_path, monkeypatch):
    """Runs the queue in ``tmp_path``, where PPO succeeds and SAC crashes.

    Args:
        run_queue (module): The ``run_queue`` module.
        runs (list[dict]): Runs of the queue.
        tmp_path (pathlib.Path): Working directory.
        monkeypatch (pytest.MonkeyPatch): Fixture used to replace the training scripts and the command line.

    Returns:
        list[list[str]]: Arguments received by the training scripts, in the order of the calls.
    """
    for name, code in [("ok.py", 0), ("crash.py", 1)]:
        (tmp_path / name).write_text(FAKE_TRAINING.format(code=code), encoding="utf-8")
    monkeypatch.setattr(run_queue, "SCRIPTS", {"ppo": str(tmp_path / "ok.py"), "sac": str(tmp_path / "crash.py")})
    queue_file = tmp_path / "queue.json"
    queue_file.write_text(json.dumps(runs), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_queue.py", str(queue_file)])
    run_queue.main()
    calls = tmp_path / "calls.jsonl"
    if not calls.exists():
        return []
    return [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines()]


def test_run_name_matches_the_results_files(run_queue):
    assert run_queue.run_name({"config": "configs/sac_trackmania.json", "seed": 2}) == "sac_trackmania_seed2"


def test_command_passes_max_steps_only_when_given(run_queue):
    base = run_queue.command({"algo": "ppo", "config": "c.json", "seed": 1})
    short = run_queue.command({"algo": "ppo", "config": "c.json", "seed": 1, "max_steps": 60000})

    assert base[0] == sys.executable
    assert base[2:] == ["--config", "c.json", "--seed", "1"]
    assert short[-2:] == ["--max-steps", "60000"]


def test_queue_goes_on_after_a_crash_and_marks_only_successes(run_queue, tmp_path, monkeypatch):
    runs = [
        {"algo": "sac", "config": "configs/a.json", "seed": 0},
        {"algo": "ppo", "config": "configs/b.json", "seed": 0, "max_steps": 60000},
    ]

    calls = start_queue(run_queue, runs, tmp_path, monkeypatch)

    assert calls == [
        ["--config", "configs/a.json", "--seed", "0"],
        ["--config", "configs/b.json", "--seed", "0", "--max-steps", "60000"],
    ]
    assert not (tmp_path / "logs" / "a_seed0.done").exists()
    assert (tmp_path / "logs" / "b_seed0.done").exists()
    assert "Pas 200 : é" in (tmp_path / "logs" / "b_seed0.log").read_text(encoding="utf-8")


def test_queue_skips_finished_runs(run_queue, tmp_path, monkeypatch):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "a_seed0.done").write_text("", encoding="utf-8")
    runs = [{"algo": "ppo", "config": "configs/a.json", "seed": 0}, {"algo": "ppo", "config": "configs/a.json", "seed": 1}]

    calls = start_queue(run_queue, runs, tmp_path, monkeypatch)

    assert calls == [["--config", "configs/a.json", "--seed", "1"]]


def test_queue_clears_the_tensorboard_folder_of_the_run(run_queue, tmp_path, monkeypatch):
    old = tmp_path / "runs" / "a_seed0" / "events.old"
    old.parent.mkdir(parents=True)
    old.write_text("", encoding="utf-8")
    kept = tmp_path / "runs" / "other_seed0"
    kept.mkdir()

    start_queue(run_queue, [{"algo": "ppo", "config": "configs/a.json", "seed": 0}], tmp_path, monkeypatch)

    assert not old.exists()
    assert kept.exists()
