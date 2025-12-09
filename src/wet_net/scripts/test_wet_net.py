import pytest
from typer.testing import CliRunner

from wet_net.scripts.wet_net import app

runner = CliRunner()


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    return data_dir


@pytest.fixture(scope="session")
def results_dir(tmp_path_factory):
    results_dir = tmp_path_factory.mktemp("results")
    return results_dir


def test_pre_process_dry_run_default():
    result = runner.invoke(app, ["pre-process", "--dry-run"])
    assert result.exit_code == 0
    assert "Would download MNIST" in result.stdout
    assert "Rerun without the --dry-run flag" in result.stdout


def test_pre_process_dry_run_custom_data_dir(data_dir):
    result = runner.invoke(app, ["pre-process", "--dry-run", "--data-dir", str(data_dir)])
    assert result.exit_code == 0
    assert "Would download MNIST" in result.stdout
    assert str(data_dir) in result.stdout


def test_train_dry_run_default():
    result = runner.invoke(app, ["train", "--dry-run"])
    assert result.exit_code == 0
    assert "Would train the model" in result.stdout
    assert "Rerun without the --dry-run flag" in result.stdout


def test_train_dry_run_with_custom_data_dir(data_dir):
    result = runner.invoke(app, ["train", "--dry-run", "--data-dir", str(data_dir)])
    assert result.exit_code == 0
    assert "Would train the model" in result.stdout
    assert str(data_dir) in result.stdout


def test_train_dry_run_with_nonexistent_data_dir():
    result = runner.invoke(app, ["train", "--dry-run", "--data-dir", "./non-existent-data"])
    assert result.exit_code == 1
    assert "does not exist" in result.stdout
    assert "non-existent-data" in result.stdout


def test_train_dry_run_with_custom_model_path():
    result = runner.invoke(app, ["train", "--dry-run", "--local-model-path", "./custom_models/model.pt"])
    assert result.exit_code == 0
    assert "Would train the model" in result.stdout
    assert "custom_models/model.pt" in result.stdout


def test_train_dry_run_with_upload_to_hub():
    result = runner.invoke(app, ["train", "--dry-run", "--upload-to-hub"])
    assert result.exit_code == 0
    assert "Would train the model" in result.stdout
    assert "Would upload the model to huggingface" in result.stdout


def test_train_dry_run_with_custom_hub_model_name():
    result = runner.invoke(
        app, ["train", "--dry-run", "--upload-to-hub", "--hub-model-name", "custom-org/custom-model"]
    )
    assert result.exit_code == 0
    assert "Would train the model" in result.stdout
    assert "custom-org/custom-model" in result.stdout


def test_train_dry_run_upload_only():
    result = runner.invoke(app, ["train", "--dry-run", "--upload-only"])
    assert result.exit_code == 0
    assert "Would upload the model" in result.stdout
    assert "Rerun without the --dry-run flag" in result.stdout


def test_train_dry_run_upload_only_with_custom_paths():
    result = runner.invoke(
        app,
        [
            "train",
            "--dry-run",
            "--upload-only",
            "--local-model-path",
            "./custom_models/model.pt",
            "--hub-model-name",
            "custom-org/custom-model",
        ],
    )
    assert result.exit_code == 0
    assert "Would upload the model" in result.stdout
    assert "custom_models/model.pt" in result.stdout
    assert "custom-org/custom-model" in result.stdout


def test_evaluate_dry_run_default():
    result = runner.invoke(app, ["evaluate", "--dry-run"])
    assert result.exit_code == 0
    assert "Would evaluate the" in result.stdout
    assert "Rerun without the --dry-run flag" in result.stdout


def test_evaluate_dry_run_with_custom_data_dir(data_dir):
    result = runner.invoke(app, ["evaluate", "--dry-run", "--data-dir", str(data_dir)])
    assert result.exit_code == 0
    assert "Would evaluate the" in result.stdout
    assert str(data_dir) in result.stdout


def test_evaluate_dry_run_with_nonexistent_data_dir():
    result = runner.invoke(app, ["evaluate", "--dry-run", "--data-dir", "./non-existent-data"])
    assert result.exit_code == 1
    assert "does not exist" in result.stdout
    assert "non-existent-data" in result.stdout


def test_evaluate_dry_run_with_custom_results_dir():
    result = runner.invoke(app, ["evaluate", "--dry-run", "--results-dir", "./custom_results"])
    assert result.exit_code == 0
    assert "Would evaluate the" in result.stdout
    assert "custom_results" in result.stdout


def test_evaluate_dry_run_with_custom_hub_model_name():
    result = runner.invoke(app, ["evaluate", "--dry-run", "--hub-model-name", "custom-org/custom-model"])
    assert result.exit_code == 0
    assert "Would evaluate the" in result.stdout
    assert "custom-org/custom-model" in result.stdout


def test_evaluate_dry_run_with_all_options(data_dir, results_dir):
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--dry-run",
            "--data-dir",
            str(data_dir),
            "--results-dir",
            str(results_dir),
            "--hub-model-name",
            "custom-org/custom-model",
        ],
    )
    assert result.exit_code == 0
    assert "Would evaluate the" in result.stdout
    assert str(data_dir) in result.stdout
    assert str(results_dir) in result.stdout
    assert "custom-org/custom-model" in result.stdout
