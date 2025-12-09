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
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True)
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
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True)
    # upload-only without artifacts may fail; we just assert it doesn't crash due to argument parsing
    assert result.returncode == 0 or result.returncode == 1
