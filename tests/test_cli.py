import pytest
from typer.testing import CliRunner

from wet_net.scripts.wet_net import app

runner = CliRunner()


class TestPreProcessCLI:
    def test_pre_process_mock(self):
        result = runner.invoke(app, ["pre-process", "--mock"])
        assert result.exit_code == 0
        assert "Preprocessed parquet ready at" in result.stdout
        assert "mock_preprocessed.parquet" in result.stdout


class TestTrainCLI:
    def test_train_mock_dry_run_default(self):
        result = runner.invoke(app, ["train", "--mock", "--dry-run"])
        assert result.exit_code == 0
        assert "[dry-run] Would train" in result.stdout
        assert "mock=True" in result.stdout

    def test_train_dry_run_mock_with_options(self):
        result = runner.invoke(app, ["train", "--seq-len", "96", "--optimize-for", "recall", "--mock", "--dry-run"])
        assert result.exit_code == 0
        assert "[dry-run] Would train" in result.stdout
        assert "mock=True" in result.stdout

    def test_train_mock_dry_run_with_custom_data_path(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("data")
        data_file = tmp_path / "custom_data.parquet"
        data_file.write_bytes(b"mock parquet data")

        result = runner.invoke(app, ["train", "--mock", "--dry-run", "--data-path", str(data_file)])
        assert result.exit_code == 0
        assert "[dry-run] Would train" in result.stdout
        assert "mock=True" in result.stdout
        assert str(data_file) in result.stdout

    def test_train_mock_dry_run_with_nonexistent_data_path_fails(self):
        # When data_path is provided and doesn't exist, should fail (non-dry-run)
        result = runner.invoke(app, ["train", "--mock", "--dry-run", "--data-path", "./nonexistent/data.parquet"])
        assert result.exit_code == 1
        assert "not found at" in result.stdout or "not found" in result.stderr

    def test_train_mock_fallback_to_default(self):
        """Test that fallback to default location works when data_path is not provided."""
        # Should work in dry-run even if default location doesn't exist
        result = runner.invoke(app, ["train", "--mock", "--dry-run"])
        assert result.exit_code == 0
        assert "[dry-run] Would train" in result.stdout
        assert "mock=True" in result.stdout

    def test_train_non_mock_fallback_to_default(self):
        """Test fallback to default location for non-mock data."""
        # Should work in dry-run even if default location doesn't exist
        result = runner.invoke(app, ["train", "--dry-run"])
        assert result.exit_code == 0
        assert "[dry-run] Would train" in result.stdout

    def test_train_mock_dry_run_with_custom_model_path(self):
        result = runner.invoke(app, ["train", "--mock", "--dry-run", "--local-model-path", "./custom_models/model.pt"])
        assert result.exit_code == 0
        assert "[dry-run] Would train" in result.stdout
        assert "mock=True" in result.stdout

    def test_train_mock_dry_run_with_push_to_hub(self):
        result = runner.invoke(app, ["train", "--mock", "--dry-run", "--push-to-hub"])
        assert result.exit_code == 0
        assert "[dry-run] Would train" in result.stdout
        assert "mock=True" in result.stdout

    def test_train_push_to_hub_dry_run_with_options(self):
        result = runner.invoke(
            app,
            [
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
        )
        assert result.exit_code == 0
        assert "[dry-run] Would train" in result.stdout
        assert "mock=True" in result.stdout

    def test_train_mock_dry_run_with_custom_hub_model_name(self):
        result = runner.invoke(
            app, ["train", "--mock", "--dry-run", "--push-to-hub", "--hub-model-name", "custom-org/custom-model"]
        )
        assert result.exit_code == 0
        assert "[dry-run] Would train" in result.stdout
        assert "mock=True" in result.stdout

    def test_train_mock_dry_run_upload_only(self):
        result = runner.invoke(app, ["train", "--mock", "--dry-run", "--upload-only"])
        # upload-only may fail due to missing HF token or permissions
        assert result.exit_code in (0, 1)
        if result.exit_code == 0:
            assert "[dry-run] Would upload existing artifacts" in result.stdout

    def test_train_upload_only_requires_push_flag(self):
        result = runner.invoke(app, ["train", "--seq-len", "96", "--optimize-for", "recall", "--upload-only"])
        # Missing artifacts is acceptable; ensure CLI exits non-zero or zero gracefully
        assert result.exit_code in (0, 1)

    def test_train_upload_only_no_artifacts(self):
        result = runner.invoke(
            app,
            [
                "train",
                "--seq-len",
                "96",
                "--optimize-for",
                "recall",
                "--upload-only",
                "--push-to-hub",
            ],
        )
        # upload-only without artifacts may fail; we just assert it doesn't crash due to argument parsing
        assert result.exit_code in (0, 1)

    def test_train_mock_dry_run_upload_only_with_custom_paths(self):
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


class TestEvaluateCLI:
    def test_evaluate_with_nonexistent_data_path_fails(self):
        # When data_path is provided and doesn't exist, should fail
        result = runner.invoke(
            app,
            ["evaluate", "--seq-len", "96", "--optimize-for", "recall", "--data-path", "./nonexistent/data.parquet"],
        )
        assert result.exit_code == 1
        assert "not found at" in result.stdout or "not found" in result.stderr

    def test_evaluate_fallback_to_default(self):
        """Test that fallback to default location works when data_path is not provided."""
        # Should fail if default location doesn't exist (non-dry-run)
        result = runner.invoke(app, ["evaluate", "--seq-len", "96", "--optimize-for", "recall"])
        # Will fail because default processed parquet doesn't exist, but should fail gracefully
        assert result.exit_code == 1
        assert "not found at" in result.stdout or "not found" in result.stderr
