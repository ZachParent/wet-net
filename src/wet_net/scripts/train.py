import shutil
from pathlib import Path

import torch
import typer
from huggingface_hub import HfApi

from wet_net.config.tri_task import SEQ_LENGTHS
from wet_net.pipelines.tri_task import train_wetnet

app = typer.Typer()


@app.command()
def train(
    seq_len: int = typer.Option(96, help="Sequence length to train (must match cached configs)."),
    optimize_for: str = typer.Option("recall", help="Optimization target: recall or false_alarm."),
    mock: bool = typer.Option(False, "--mock", help="Use mock dataset (requires pre_process --mock)."),
    data_path: str | None = typer.Option(None, help="Preprocessed parquet path; defaults to processed output."),
    local_model_path: str | None = typer.Option(None, help="Optional path to copy wetnet.pt after training."),
    dry_run: bool = typer.Option(False, help="Show what would happen without training."),
    push_to_hub: bool = typer.Option(False, help="Upload artifacts to Hugging Face after training."),
    upload_only: bool = typer.Option(
        False, help="Skip training; just upload existing artifacts (implies --push-to-hub)."
    ),
    hub_model_name: str = typer.Option("WetNet/wet-net", help="Repo name to push to on Hugging Face."),
    seed: int = typer.Option(42, help="Random seed for reproducibility (matches notebook defaults)."),
    push_model_only: bool = typer.Option(
        False, help="When pushing to hub, upload only wetnet.pt (skip VIB/config/metrics)."
    ),
    fast: bool = typer.Option(False, help="Enable fast mode: cap stage/VIB epochs for quick iteration."),
    max_epochs: int | None = typer.Option(
        None,
        help="Optional hard cap on epochs per stage (overrides fast default). Use small number (e.g., 2).",
    ),
    no_early_stop: bool = typer.Option(
        False, help="Run full planned epochs per stage (match notebook), disable early stopping."
    ),
    min_delta_abs: float = typer.Option(
        1e-4,
        help="Absolute improvement required to reset patience (avoid tiny val losses triggering early stop).",
    ),
    min_delta_rel: float = typer.Option(
        0.0,
        help="Relative improvement (fraction of best val loss) required to reset patience.",
    ),
    min_anomaly_ratio: float = typer.Option(
        0.0,
        help="Minimum anomaly fraction enforced in val/test splits (0.0 keeps current behavior; e.g., 0.05 = 5%).",
    ),
    recon_weight: float = typer.Option(1.0, help="Loss weight for reconstruction task."),
    forecast_weight: float = typer.Option(0.6, help="Loss weight for forecast task."),
    short_weight: float = typer.Option(1.2, help="Loss weight for short-horizon anomaly task."),
    long_weight: float = typer.Option(1.2, help="Loss weight for long-horizon anomaly task."),
    early_stop_metric: str = typer.Option(
        "total",
        help="Metric to monitor for early stopping: total (all losses) or cls (classification heads only).",
        case_sensitive=False,
    ),
    run_suffix: str = typer.Option(
        "",
        help="Optional suffix for run directory (e.g., _fast) to avoid overwriting full runs.",
    ),
    results_dir: str = typer.Option("./results", help="Directory to save training results."),
):
    """
    Train WetNet with the cached best configuration (no grid search).
    """
    if seq_len not in SEQ_LENGTHS:
        typer.secho(f"seq_len must be one of {SEQ_LENGTHS}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    upload_to_hub = push_to_hub or upload_only
    suffix = run_suffix or ("_fast" if fast or max_epochs else "")
    run_id = f"seq{seq_len}_{optimize_for}{suffix}"
    results_dir_path = Path(results_dir)
    base_dir = results_dir_path / "wetnet" / run_id

    # If user only wants to push the existing model, skip training when artifact is present.
    if push_model_only and not upload_only:
        model_candidate = base_dir / "wetnet.pt"
        if model_candidate.exists():
            upload_only = True
            upload_to_hub = True
            typer.secho(
                f"Model artifact found at {model_candidate}. Skipping training and pushing model only.",
                fg=typer.colors.YELLOW,
            )
        else:
            typer.secho(
                f"No existing model at {model_candidate}; training will run then push model only.",
                fg=typer.colors.YELLOW,
            )

    artifacts = None
    task_weights = {
        "reconstruction": recon_weight,
        "forecast": forecast_weight,
        "short": short_weight,
        "long": long_weight,
    }
    monitor = early_stop_metric.lower()
    if monitor not in {"total", "cls"}:
        typer.secho("early_stop_metric must be one of: total, cls", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if upload_only and dry_run:
        typer.secho(
            f"[dry-run] Would upload existing artifacts to Hugging Face hub_model_name={hub_model_name}.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=0)

    if not upload_only:
        # Locate preprocessed parquet only when training is needed.
        # If data_path is explicitly provided, it must exist (no fallback).
        if data_path:
            data_path_obj = Path(data_path)
            if not data_path_obj.exists():
                typer.secho(
                    f"Preprocessed parquet not found at {data_path_obj}. "
                    f"Run `wet-net pre-process {'--mock' if mock else '--data-url <url>'}` first.",
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=1)
            preprocessed = data_path_obj
        else:
            # Fallback to default locations if data_path not provided
            if mock:
                # Default mock location relative to current directory
                preprocessed = Path("./data/processed/mock_preprocessed.parquet")
            else:
                from wet_net.data.preprocess import PROCESSED_PARQUET

                preprocessed = PROCESSED_PARQUET
            if not preprocessed.exists() and not dry_run:
                msg = (
                    "Mock preprocessed parquet not found. Run `wet-net pre-process --mock` first."
                    if mock
                    else "Preprocessed parquet not found. Run `wet-net pre-process --data-url <url>` first."
                )
                typer.secho(msg, fg=typer.colors.RED)
                raise typer.Exit(code=1)
            if not preprocessed.exists():
                preprocessed = Path("dry-run-placeholder")

        if dry_run:
            typer.secho(
                f"[dry-run] Would train seq_len={seq_len}, optimize_for={optimize_for}, "
                f"mock={mock}, preprocessed={preprocessed}, hub_model_name={hub_model_name}",
                fg=typer.colors.YELLOW,
            )
            raise typer.Exit(code=0)

        artifacts = train_wetnet(
            seq_len=seq_len,
            optimize_for=optimize_for,
            preprocessed_path=preprocessed,
            device=device,
            results_dir=results_dir_path,
            mock=mock,
            seed=seed,
            fast_epochs=(max_epochs if max_epochs is not None else (2 if fast else None)),
            run_suffix=suffix,
            early_stop=not no_early_stop,
            min_delta_abs=min_delta_abs,
            min_delta_rel=min_delta_rel,
            min_anomaly_ratio=min_anomaly_ratio,
            task_weights=task_weights,
            monitor=monitor,
        )
        typer.secho(f"Training complete. Saved artifacts to {artifacts['model'].parent}", fg=typer.colors.GREEN)
        if local_model_path:
            dst = Path(local_model_path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(artifacts["model"], dst)
            typer.secho(f"Copied model to {dst}", fg=typer.colors.GREEN)
    else:
        artifacts = {
            "model": base_dir / "wetnet.pt",
            "vib": base_dir / "vib.pt",
            "config": base_dir / "config.json",
            "metrics": base_dir / "metrics.csv",
            "augmented_metrics": base_dir / "augmented_metrics.csv",
        }

    if upload_to_hub:
        push_artifacts_to_hub(
            artifacts=artifacts,
            hub_model_name=hub_model_name,
            run_prefix=f"seq{seq_len}_{optimize_for}{suffix}",
            push_model_only=push_model_only,
        )


def push_artifacts_to_hub(
    artifacts: dict[str, Path],
    hub_model_name: str,
    run_prefix: str,
    push_model_only: bool,
) -> None:
    api = HfApi()
    repo_id = hub_model_name
    typer.secho(f"Pushing artifacts to Hugging Face repo {repo_id} ...", fg=typer.colors.YELLOW)
    api.create_repo(repo_id=repo_id, exist_ok=True)
    keys_to_push = ["model"] if push_model_only else ["model", "vib", "config", "metrics", "augmented_metrics"]
    for key in keys_to_push:
        path = artifacts.get(key)
        if path and Path(path).exists():
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=str(Path(run_prefix) / Path(path).name),
                repo_id=repo_id,
            )
    typer.secho("Push complete.", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
