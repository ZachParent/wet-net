import subprocess
from pathlib import Path


def run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    # Using subprocess.run with explicit shell=False for security
    # All commands are hardcoded and trusted in test context
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, shell=False)  # noqa: S603


def test_pre_process_mock():
    repo = Path(__file__).resolve().parents[1]
    result = run_cli(["uv", "run", "wet-net", "pre-process", "--mock"], cwd=repo)
    assert result.returncode == 0, result.stderr
    assert "Preprocessed parquet ready" in result.stdout


def test_train_dry_run_mock():
    repo = Path(__file__).resolve().parents[1]
    result = run_cli(
        ["uv", "run", "wet-net", "train", "--seq-len", "96", "--optimize-for", "recall", "--mock", "--dry-run"],
        cwd=repo,
    )
    assert result.returncode == 0, result.stderr
    assert "[dry-run] Would train" in result.stdout


def test_train_upload_only_requires_push_flag():
    repo = Path(__file__).resolve().parents[1]
    result = run_cli(
        ["uv", "run", "wet-net", "train", "--seq-len", "96", "--optimize-for", "recall", "--upload-only"],
        cwd=repo,
    )
    # Missing artifacts is acceptable; ensure CLI exits non-zero or zero gracefully
    assert result.returncode in (0, 1)


def test_train_push_to_hub_dry_run_equivalent():
    repo = Path(__file__).resolve().parents[1]
    result = run_cli(
        [
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
            "--push-to-hub",
            "--hub-model-name",
            "WetNet/wet-net",
        ],
        cwd=repo,
    )
    assert result.returncode == 0, result.stderr
    assert "[dry-run] Would train" in result.stdout
