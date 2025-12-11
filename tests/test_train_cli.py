import subprocess
from pathlib import Path


def test_train_dry_run_mock():
    cmd = [
        "uv",
        "run",
        "wet-net",
        "train",
        "--seq-len",
        "96",
        "--optimize-for",
        "recall",
        "--mock",
        "--dry-run",
    ]
    # Using subprocess.run with explicit shell=False for security
    # All commands are hardcoded and trusted in test context
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True, shell=False)  # noqa: S603
    assert result.returncode == 0, result.stderr.decode()


def test_train_upload_only_no_artifacts():
    cmd = [
        "uv",
        "run",
        "wet-net",
        "train",
        "--seq-len",
        "96",
        "--optimize-for",
        "recall",
        "--upload-only",
        "--push-to-hub",
    ]
    # Using subprocess.run with explicit shell=False for security
    # All commands are hardcoded and trusted in test context
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True, shell=False)  # noqa: S603
    # upload-only without artifacts may fail; we just assert it doesn't crash due to argument parsing
    assert result.returncode == 0 or result.returncode == 1
