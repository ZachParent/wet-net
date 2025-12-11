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
        assert "hub_model_name=WetNet/wet-net" in result.stdout

    def test_train_dry_run_mock_with_options(self):
        result = runner.invoke(app, ["train", "--seq-len", "96", "--optimize-for", "recall", "--mock", "--dry-run"])
        assert result.exit_code == 0
        assert "[dry-run] Would train" in result.stdout
        assert "mock=True" in result.stdout
        assert "hub_model_name=WetNet/wet-net" in result.stdout

    def test_train_mock_dry_run_with_custom_data_path(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("data")
        data_file = tmp_path / "custom_data.parquet"
        data_file.write_bytes(b"mock parquet data")

        result = runner.invoke(app, ["train", "--mock", "--dry-run", "--data-path", str(data_file)])
        assert result.exit_code == 0
        assert "[dry-run] Would train" in result.stdout
        assert "mock=True" in result.stdout
        assert str(data_file) in result.stdout
        assert "hub_model_name=WetNet/wet-net" in result.stdout

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
        assert "hub_model_name=WetNet/wet-net" in result.stdout

    def test_train_non_mock_fallback_to_default(self):
        """Test fallback to default location for non-mock data."""
        # Should work in dry-run even if default location doesn't exist
        result = runner.invoke(app, ["train", "--dry-run"])
        assert result.exit_code == 0
        assert "[dry-run] Would train" in result.stdout
        assert "hub_model_name=WetNet/wet-net" in result.stdout

    def test_train_mock_dry_run_with_custom_model_path(self):
        result = runner.invoke(app, ["train", "--mock", "--dry-run", "--local-model-path", "./custom_models/model.pt"])
        assert result.exit_code == 0
        assert "[dry-run] Would train" in result.stdout
        assert "mock=True" in result.stdout
        assert "hub_model_name=WetNet/wet-net" in result.stdout

    def test_train_mock_dry_run_with_push_to_hub(self):
        result = runner.invoke(app, ["train", "--mock", "--dry-run", "--push-to-hub"])
        assert result.exit_code == 0
        assert "[dry-run] Would train" in result.stdout
        assert "mock=True" in result.stdout
        assert "hub_model_name=WetNet/wet-net" in result.stdout

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
        assert "hub_model_name=WetNet/wet-net" in result.stdout

    def test_train_mock_dry_run_with_custom_hub_model_name(self):
        result = runner.invoke(
            app, ["train", "--mock", "--dry-run", "--push-to-hub", "--hub-model-name", "custom-org/custom-model"]
        )
        assert result.exit_code == 0
        assert "[dry-run] Would train" in result.stdout
        assert "mock=True" in result.stdout
        assert "hub_model_name=custom-org/custom-model" in result.stdout

    def test_train_mock_dry_run_upload_only(self):
        result = runner.invoke(app, ["train", "--mock", "--dry-run", "--upload-only"])
        # upload-only with dry-run should exit early with specific message
        assert result.exit_code == 0
        assert "[dry-run] Would upload existing artifacts to Hugging Face" in result.stdout
        assert "hub_model_name=WetNet/wet-net" in result.stdout

    def test_train_mock(self, tmp_path_factory):
        """Test full training pipeline with mock data."""
        # First ensure mock data exists
        preprocess_result = runner.invoke(app, ["pre-process", "--mock"])
        assert preprocess_result.exit_code == 0

        # Use a temporary directory for results
        tmp_results = tmp_path_factory.mktemp("results")
        results_dir = str(tmp_results)

        result = runner.invoke(
            app,
            [
                "train",
                "--seq-len",
                "96",
                "--optimize-for",
                "recall",
                "--mock",
                "--fast",
                "--max-epochs",
                "1",
                "--results-dir",
                results_dir,
            ],
        )
        assert result.exit_code == 0
        assert "Training complete" in result.stdout
        assert "Saved artifacts to" in result.stdout

        # Verify artifacts were created using the same results_dir
        from pathlib import Path

        artifacts_dir = Path(results_dir) / "wetnet" / "seq96_recall_fast"
        assert (artifacts_dir / "wetnet.pt").exists()
        assert (artifacts_dir / "vib.pt").exists()
        assert (artifacts_dir / "config.json").exists()
        assert (artifacts_dir / "metrics.csv").exists()
        assert (artifacts_dir / "augmented_metrics.csv").exists()

    def test_train_mock_with_local_model_path(self, tmp_path_factory):
        """Test training with custom local model path."""
        # First ensure mock data exists
        preprocess_result = runner.invoke(app, ["pre-process", "--mock"])
        assert preprocess_result.exit_code == 0

        tmp_path = tmp_path_factory.mktemp("models")
        model_path = tmp_path / "custom_model.pt"

        result = runner.invoke(
            app,
            [
                "train",
                "--seq-len",
                "96",
                "--optimize-for",
                "recall",
                "--mock",
                "--fast",
                "--max-epochs",
                "1",
                "--local-model-path",
                str(model_path),
            ],
        )
        assert result.exit_code == 0
        assert "Training complete" in result.stdout
        assert "Copied model to" in result.stdout
        assert str(model_path) in result.stdout
        assert model_path.exists()

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
        # upload-only with dry-run should exit early with specific message
        assert result.exit_code == 0
        assert "[dry-run] Would upload existing artifacts to Hugging Face" in result.stdout
        assert "hub_model_name=custom-org/custom-model" in result.stdout


class TestEvaluateCLI:
    def test_evaluate_with_nonexistent_data_path_fails(self):
        # When data_path is provided and doesn't exist, should fail
        result = runner.invoke(
            app,
            ["evaluate", "--seq-len", "96", "--optimize-for", "recall", "--data-path", "./nonexistent/data.parquet"],
        )
        assert result.exit_code == 1
        assert "not found at" in result.stdout or "not found" in result.stderr

    def test_evaluate_mock_end_to_end(self, tmp_path_factory):
        """Test full evaluation pipeline with mock data using default behavior."""
        # First ensure mock data exists
        preprocess_result = runner.invoke(app, ["pre-process", "--mock"])
        assert preprocess_result.exit_code == 0

        # Use a temporary directory for results
        tmp_results = tmp_path_factory.mktemp("results")
        results_dir = str(tmp_results)

        # Train a model first (with fast mode)
        train_result = runner.invoke(
            app,
            [
                "train",
                "--seq-len",
                "96",
                "--optimize-for",
                "recall",
                "--mock",
                "--fast",
                "--max-epochs",
                "1",
                "--results-dir",
                results_dir,
            ],
        )
        assert train_result.exit_code == 0

        # Now evaluate using the trained model (default behavior - checks local training dir)
        from pathlib import Path

        mock_data_path = Path("./data/processed/mock_preprocessed.parquet")
        result = runner.invoke(
            app,
            [
                "evaluate",
                "--seq-len",
                "96",
                "--optimize-for",
                "recall",
                "--data-path",
                str(mock_data_path),
                "--run-suffix",
                "_fast",
                "--results-dir",
                results_dir,
            ],
        )
        assert result.exit_code == 0
        assert "Report written to" in result.stdout

        # Verify report artifacts were created using the same results_dir
        report_dir = Path(results_dir) / "wetnet" / "report" / "seq96_recall_fast"
        assert (report_dir / "report.md").exists()
        assert (report_dir / "metrics.csv").exists()
        assert (report_dir / "augmented_metrics.csv").exists()
        assert (report_dir / "threshold_sweep.csv").exists()

    def test_evaluate_with_local_artifacts_path_success(self, tmp_path_factory):
        """Test evaluation with --local-artifacts-path pointing to existing artifacts."""
        # First ensure mock data exists
        preprocess_result = runner.invoke(app, ["pre-process", "--mock"])
        assert preprocess_result.exit_code == 0

        # Use a temporary directory for results
        tmp_results = tmp_path_factory.mktemp("results")
        results_dir = str(tmp_results)

        # Train a model first (with fast mode)
        train_result = runner.invoke(
            app,
            [
                "train",
                "--seq-len",
                "96",
                "--optimize-for",
                "recall",
                "--mock",
                "--fast",
                "--max-epochs",
                "1",
                "--results-dir",
                results_dir,
            ],
        )
        assert train_result.exit_code == 0

        # Get the path to the trained artifacts
        from pathlib import Path

        artifacts_dir = Path(results_dir) / "wetnet" / "seq96_recall_fast"
        mock_data_path = Path("./data/processed/mock_preprocessed.parquet")

        # Evaluate using --local-artifacts-path
        result = runner.invoke(
            app,
            [
                "evaluate",
                "--seq-len",
                "96",
                "--optimize-for",
                "recall",
                "--data-path",
                str(mock_data_path),
                "--local-artifacts-path",
                str(artifacts_dir),
                "--results-dir",
                results_dir,
            ],
        )
        assert result.exit_code == 0
        assert "Report written to" in result.stdout

        # Verify report artifacts were created using the same results_dir
        report_dir = Path(results_dir) / "wetnet" / "report" / "seq96_recall"
        assert (report_dir / "report.md").exists()
        assert (report_dir / "metrics.csv").exists()

    def test_evaluate_with_local_artifacts_path_fails_nonexistent(self):
        """Test that --local-artifacts-path fails when path doesn't exist."""
        from pathlib import Path

        mock_data_path = Path("./data/processed/mock_preprocessed.parquet")
        # Ensure mock data exists
        if not mock_data_path.exists():
            runner.invoke(app, ["pre-process", "--mock"])

        result = runner.invoke(
            app,
            [
                "evaluate",
                "--seq-len",
                "96",
                "--optimize-for",
                "recall",
                "--data-path",
                str(mock_data_path),
                "--local-artifacts-path",
                "/nonexistent/path/to/artifacts",
                "--results-dir",
                "./results",
            ],
        )
        assert result.exit_code == 1
        assert "does not exist" in result.stdout or "does not exist" in result.stderr

    def test_evaluate_with_local_artifacts_path_fails_missing_artifacts(self, tmp_path_factory):
        """Test that --local-artifacts-path fails when required artifacts are missing."""
        from pathlib import Path

        mock_data_path = Path("./data/processed/mock_preprocessed.parquet")
        # Ensure mock data exists
        if not mock_data_path.exists():
            runner.invoke(app, ["pre-process", "--mock"])

        # Create a temporary directory that exists but doesn't have the required artifacts
        tmp_dir = tmp_path_factory.mktemp("empty_artifacts")
        result = runner.invoke(
            app,
            [
                "evaluate",
                "--seq-len",
                "96",
                "--optimize-for",
                "recall",
                "--data-path",
                str(mock_data_path),
                "--local-artifacts-path",
                str(tmp_dir),
                "--results-dir",
                "./results",
            ],
        )
        assert result.exit_code == 1
        assert "missing" in result.stdout.lower() or "missing" in result.stderr.lower()

    def test_evaluate_with_hub_model_name_fails_nonexistent(self):
        """Test that --hub-model-name fails when repo doesn't exist."""
        from pathlib import Path

        mock_data_path = Path("./data/processed/mock_preprocessed.parquet")
        # Ensure mock data exists
        if not mock_data_path.exists():
            runner.invoke(app, ["pre-process", "--mock"])

        result = runner.invoke(
            app,
            [
                "evaluate",
                "--seq-len",
                "96",
                "--optimize-for",
                "recall",
                "--data-path",
                str(mock_data_path),
                "--hub-model-name",
                "nonexistent-org/nonexistent-model-12345",
                "--results-dir",
                "./results",
            ],
        )
        assert result.exit_code == 1
        # Should fail with repository not found error
        assert "404" in result.stdout or "not found" in result.stdout.lower() or "not found" in result.stderr.lower()

    def test_evaluate_with_both_local_and_hub_fails(self):
        """Test that specifying both --local-artifacts-path and --hub-model-name fails."""
        from pathlib import Path

        mock_data_path = Path("./data/processed/mock_preprocessed.parquet")
        # Ensure mock data exists
        if not mock_data_path.exists():
            runner.invoke(app, ["pre-process", "--mock"])

        result = runner.invoke(
            app,
            [
                "evaluate",
                "--seq-len",
                "96",
                "--optimize-for",
                "recall",
                "--data-path",
                str(mock_data_path),
                "--local-artifacts-path",
                "/some/path",
                "--hub-model-name",
                "some-org/some-model",
                "--results-dir",
                "./results",
            ],
        )
        assert result.exit_code == 1
        assert "both" in result.stdout.lower() or "both" in result.stderr.lower()

    def test_evaluate_default_uses_hub_when_local_missing(self, tmp_path_factory):
        """Test that default evaluate behavior uses hub when local artifacts don't exist."""
        from pathlib import Path

        mock_data_path = Path("./data/processed/mock_preprocessed.parquet")
        # Ensure mock data exists
        if not mock_data_path.exists():
            runner.invoke(app, ["pre-process", "--mock"])

        # Use a temporary directory for results
        tmp_results = tmp_path_factory.mktemp("results")
        results_dir = str(tmp_results)

        # Use a run_id that we know doesn't exist locally to ensure hub is used
        # Using a unique run_id that won't conflict with other tests
        run_id = "seq96_recall_hub_test"
        local_training_dir = Path(results_dir) / "wetnet" / run_id
        # Verify local doesn't exist (if it does, something is wrong)
        assert not local_training_dir.exists(), f"Local directory {local_training_dir} should not exist for this test"

        # Call evaluate without --local-artifacts-path or --hub-model-name
        # This should trigger Priority 3: check local (not found), then fallback to hub
        result = runner.invoke(
            app,
            [
                "evaluate",
                "--seq-len",
                "96",
                "--optimize-for",
                "recall",
                "--data-path",
                str(mock_data_path),
                "--run-suffix",
                "_hub_test",
                "--results-dir",
                results_dir,
            ],
        )

        # Should successfully download from hub - verify report shows hub source
        assert result.exit_code == 0, f"Evaluate failed: {result.stdout}\n{result.stderr}"
        assert "Report written to" in result.stdout
        report_dir = Path(results_dir) / "wetnet" / "report" / run_id
        report_path = report_dir / "report.md"
        assert report_path.exists(), "Report file should be created"
        report_content = report_path.read_text()
        # Verify it used the default hub repo
        assert "Hub: WetNet/wet-net (default)" in report_content
