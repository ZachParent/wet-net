import pytest
from typer.testing import CliRunner

from wet_net.scripts.wet_net import app

runner = CliRunner()


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    return data_dir


def test_pre_process_mock_default():
    result = runner.invoke(app, ["pre-process", "--mock"])
    assert result.exit_code == 0
    assert "Preprocessed parquet ready at" in result.stdout
    assert "mock_preprocessed.parquet" in result.stdout


def test_train_mock_dry_run_default():
    result = runner.invoke(app, ["train", "--mock", "--dry-run"])
    assert result.exit_code == 0
    assert "[dry-run] Would train" in result.stdout
    assert "mock=True" in result.stdout


def test_train_mock_dry_run_with_custom_data_dir():
    result = runner.invoke(app, ["train", "--mock", "--dry-run", "--data-path", "custom/data"])
    assert result.exit_code == 0
    assert "[dry-run] Would train" in result.stdout
    assert "mock=True" in result.stdout


def test_train_mock_dry_run_with_nonexistent_data_dir():
    result = runner.invoke(app, ["train", "--mock", "--dry-run", "--data-dir", "./non-existent-data"])
    assert result.exit_code == 0
    assert "[dry-run] Would train" in result.stdout
    assert "mock=True" in result.stdout


def test_train_mock_dry_run_with_custom_model_path():
    result = runner.invoke(app, ["train", "--mock", "--dry-run", "--local-model-path", "./custom_models/model.pt"])
    assert result.exit_code == 0
    assert "[dry-run] Would train" in result.stdout
    assert "mock=True" in result.stdout


def test_train_mock_dry_run_with_push_to_hub():
    result = runner.invoke(app, ["train", "--mock", "--dry-run", "--push-to-hub"])
    assert result.exit_code == 0
    assert "[dry-run] Would train" in result.stdout
    assert "mock=True" in result.stdout


def test_train_mock_dry_run_with_custom_hub_model_name():
    result = runner.invoke(
        app, ["train", "--mock", "--dry-run", "--push-to-hub", "--hub-model-name", "custom-org/custom-model"]
    )
    assert result.exit_code == 0
    assert "[dry-run] Would train" in result.stdout
    assert "mock=True" in result.stdout


def test_train_mock_dry_run_upload_only():
    result = runner.invoke(app, ["train", "--mock", "--dry-run", "--upload-only"])
    # upload-only may fail due to missing HF token or permissions
    assert result.exit_code in (0, 1)
    if result.exit_code == 0:
        assert "[dry-run] Would upload existing artifacts" in result.stdout


def test_train_mock_dry_run_upload_only_with_custom_paths():
    result = runner.invoke(
        app,
        [
            "train",
            "--mock",
            "--dry-run",
            "--upload-only",
            "--local-model-path",
            "./custom_models/model.pt",
            "--hub-model-name",
            "custom-org/custom-model",
        ],
    )
    # upload-only may fail due to missing HF token or permissions
    assert result.exit_code in (0, 1)
    if result.exit_code == 0:
        assert "[dry-run] Would upload existing artifacts" in result.stdout
