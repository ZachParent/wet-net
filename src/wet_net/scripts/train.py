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
    data_dir: str = typer.Option(
        "./data", help="(legacy) data directory; used to locate preprocessed parquet if provided."
    ),
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
):
    """
    Train WetNet with the cached best configuration (no grid search).
    """
    if seq_len not in SEQ_LENGTHS:
        typer.secho(f"seq_len must be one of {SEQ_LENGTHS}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    upload_to_hub = push_to_hub or upload_only
    run_id = f"seq{seq_len}_{optimize_for}"
    base_dir = Path("results/wetnet") / run_id

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
    if not upload_only:
        # Locate preprocessed parquet only when training is needed.
        candidates = []
        if data_path:
            candidates.append(Path(data_path))
        if mock:
            from wet_net.data.preprocess import DATA_DIR
            candidates.append(DATA_DIR / "processed" / "mock_preprocessed.parquet")
            candidates.append(Path(data_dir) / "processed" / "mock_preprocessed.parquet")
        else:
            from wet_net.data.preprocess import PROCESSED_PARQUET
            candidates.append(PROCESSED_PARQUET)
            candidates.append(Path(data_dir) / "processed" / "anomalous_consumption_preprocessed.parquet")
        preprocessed = next((c for c in candidates if c and c.exists()), None)
        if preprocessed is None and not dry_run:
            msg = (
                "Mock preprocessed parquet not found. Run `wet-net pre-process --mock` first."
                if mock
                else "Preprocessed parquet not found. Run `wet-net pre-process --data-url <url>` first."
            )
            typer.secho(msg, fg=typer.colors.RED)
            raise typer.Exit(code=1)
        preprocessed = Path(preprocessed) if preprocessed else Path("dry-run-placeholder")

        if dry_run:
            typer.secho(
                f"[dry-run] Would train seq_len={seq_len}, optimize_for={optimize_for}, "
                f"mock={mock}, preprocessed={preprocessed}",
                fg=typer.colors.YELLOW,
            )
            if upload_only:
                typer.secho("[dry-run] Would upload existing artifacts to Hugging Face.", fg=typer.colors.YELLOW)
            raise typer.Exit(code=0)

        artifacts = train_wetnet(
            seq_len=seq_len,
            optimize_for=optimize_for,
            preprocessed_path=preprocessed,
            device=device,
            mock=mock,
            seed=seed,
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
        api = HfApi()
        repo_id = hub_model_name
        typer.secho(f"Pushing artifacts to Hugging Face repo {repo_id} ...", fg=typer.colors.YELLOW)
        api.create_repo(repo_id=repo_id, exist_ok=True)
        run_prefix = f"seq{seq_len}_{optimize_for}"
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
