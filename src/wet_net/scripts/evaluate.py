"""
End-to-end evaluation and reporting for WetNet.

Features:
- Downloads model/vib/config from a Hugging Face repo if not already present.
- Rebuilds the test split, runs inference, sweeps thresholds, and computes metrics.
- Generates plots inspired by notebooks 07_tri_task_nexus and 08_conflict_top5_analysis:
  * Conflict histogram (anomalous vs normal)
  * Threshold sweep bar chart
  * Forecast vs future example plot
- Writes a concise markdown report plus CSV tables.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import typer
from huggingface_hub import snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError

from wet_net.config.tri_task import HORIZONS, METRIC_THRESHOLDS, SEQ_LENGTHS, get_best_config
from wet_net.data.datasets import (
    TimeSeriesDataset,
    TriTaskWindowDataset,
    build_metadata,
    build_policy_split,
    compute_future_sequences,
    ensure_anomaly_coverage,
    make_dataloaders,
)
from wet_net.data.preprocess import load_preprocessed_dataframe, select_feature_columns
from wet_net.eval.metrics import evaluate_multi_horizon, sweep_fusion_thresholds
from wet_net.eval.predictions import build_prediction_frame, collect_predictions
from wet_net.models.vib import VIBTransformer
from wet_net.models.wetnet import WetNet
from wet_net.training.fusion import fuse_probabilities
from wet_net.training.utils import (
    batch_for_seq,
    intelligent_batch_size,
    max_samples_for_seq,
    set_seed,
    stride_for_seq,
)

app = typer.Typer()


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def ensure_artifacts(
    run_id: str,
    out_dir: Path,
    results_dir: Path,
    local_artifacts_path: Path | None = None,
    hub_model_name: str | None = None,
) -> dict[str, Path]:
    """
    Ensure artifacts exist locally.

    Priority order:
    1. If local_artifacts_path is provided, use it (fail if missing)
    2. If hub_model_name is provided, download from HuggingFace Hub (fail if missing)
    3. Otherwise, check local training directory, then fallback to default hub repo
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = {
        "model": out_dir / "wetnet.pt",
        "vib": out_dir / "vib.pt",
        "config": out_dir / "config.json",
    }

    # Priority 1: Use explicitly provided local artifacts path
    if local_artifacts_path is not None:
        local_artifacts_path_obj = Path(local_artifacts_path)
        if not local_artifacts_path_obj.exists():
            raise FileNotFoundError(f"Local artifacts path does not exist: {local_artifacts_path_obj}")

        local_artifacts = {
            "model": local_artifacts_path_obj / "wetnet.pt",
            "vib": local_artifacts_path_obj / "vib.pt",
            "config": local_artifacts_path_obj / "config.json",
        }

        # Verify all required artifacts exist
        missing = [name for name, path in local_artifacts.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Required artifacts missing from {local_artifacts_path_obj}: {', '.join(missing)}")

        # Copy artifacts to out_dir
        for key, local_path in local_artifacts.items():
            expected[key].write_bytes(local_path.read_bytes())

        # Copy optional CSVs if present
        for fname in ["metrics.csv", "augmented_metrics.csv"]:
            src = local_artifacts_path_obj / fname
            if src.exists():
                dst = out_dir / fname
                if not dst.exists():
                    dst.write_bytes(src.read_bytes())
        return expected

    # Priority 2: Download from explicitly provided hub model name
    if hub_model_name is not None:
        snapshot_dir = Path(
            snapshot_download(
                hub_model_name,
                allow_patterns=["wetnet.pt", "vib.pt", "config.json", "metrics.csv", "augmented_metrics.csv"],
            )
        )
        # Look for artifacts in the run_id subdirectory
        run_dir = snapshot_dir / run_id
        artifact_dir = run_dir if run_dir.exists() else snapshot_dir

        for _name, path in expected.items():
            src = artifact_dir / path.name
            if not src.exists():
                raise FileNotFoundError(f"{path.name} not found in repo {hub_model_name} (checked {artifact_dir})")
            path.write_bytes(src.read_bytes())

        # Copy optional CSVs if present
        for fname in ["metrics.csv", "augmented_metrics.csv"]:
            src = artifact_dir / fname
            if src.exists():
                dst = out_dir / fname
                if not dst.exists():
                    dst.write_bytes(src.read_bytes())
        return expected

    # Priority 3: Check local training directory first, then fallback to default hub repo
    local_training_dir = results_dir / "wetnet" / run_id
    local_artifacts = {
        "model": local_training_dir / "wetnet.pt",
        "vib": local_training_dir / "vib.pt",
        "config": local_training_dir / "config.json",
    }

    # If all artifacts exist locally, use them (copy to out_dir for consistency)
    if all(p.exists() for p in local_artifacts.values()):
        # Copy artifacts to out_dir
        for key, local_path in local_artifacts.items():
            expected[key].write_bytes(local_path.read_bytes())

        # Copy optional CSVs if present
        for fname in ["metrics.csv", "augmented_metrics.csv"]:
            src = local_training_dir / fname
            if src.exists():
                dst = out_dir / fname
                if not dst.exists():
                    dst.write_bytes(src.read_bytes())
        return expected

    # If not found locally, check out_dir
    if all(p.exists() for p in expected.values()):
        return expected

    # Finally, try downloading from default Hugging Face repo
    default_repo_id = "WetNet/wet-net"
    snapshot_dir = Path(
        snapshot_download(
            default_repo_id,
            allow_patterns=["wetnet.pt", "vib.pt", "config.json", "metrics.csv", "augmented_metrics.csv"],
        )
    )
    # Look for artifacts in the run_id subdirectory
    run_dir = snapshot_dir / run_id
    artifact_dir = run_dir if run_dir.exists() else snapshot_dir

    for _name, path in expected.items():
        src = artifact_dir / path.name
        if not src.exists():
            raise FileNotFoundError(f"{path.name} not found in repo {default_repo_id} (checked {artifact_dir})")
        path.write_bytes(src.read_bytes())

    # Copy optional CSVs if present
    for fname in ["metrics.csv", "augmented_metrics.csv"]:
        src = artifact_dir / fname
        if src.exists():
            dst = out_dir / fname
            if not dst.exists():
                dst.write_bytes(src.read_bytes())
    return expected


def build_loaders(preprocessed: Path, seq_len: int):
    df = load_preprocessed_dataframe(preprocessed)
    feature_cols = select_feature_columns(df)
    stride = stride_for_seq(seq_len)
    max_samples = max_samples_for_seq(seq_len)
    base_dataset = TimeSeriesDataset(
        df,
        seq_len=seq_len,
        horizons=HORIZONS,
        stride=stride,
        max_samples=max_samples,
        feature_cols=feature_cols,
    )
    future_targets, anchors, policies = compute_future_sequences(df, base_dataset, forecast_horizon=24)
    tri_dataset = TriTaskWindowDataset(base_dataset, future_targets)
    metadata = build_metadata(base_dataset, anchors, policies, HORIZONS)
    splits = build_policy_split(metadata, (0.7, 0.15, 0.15))
    ensure_anomaly_coverage(metadata, splits)
    base_batch = batch_for_seq(seq_len)
    batch_size = intelligent_batch_size(seq_len, len(feature_cols), base_batch, d_model_guess=256)
    loaders = make_dataloaders(tri_dataset, splits, batch_size)
    return df, tri_dataset, metadata, splits, loaders, feature_cols


def threshold_curves(probs: np.ndarray, labels: np.ndarray, thr_grid: Iterable[float]) -> pd.DataFrame:
    rows = []
    for thr in thr_grid:
        alert = probs >= thr
        recall = float(alert[labels == 1].mean()) if np.any(labels == 1) else np.nan
        false_alarm = float(alert[labels == 0].mean()) if np.any(labels == 0) else np.nan
        rows.append({"threshold": thr, "recall": recall, "false_alarm": false_alarm})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plotting                                                                    #
# --------------------------------------------------------------------------- #


def plot_conflict(conflict_scores: np.ndarray, labels: np.ndarray, out_dir: Path) -> Path:
    sns.set_style("whitegrid")
    df = pd.DataFrame({"conflict": conflict_scores, "label": np.where(labels == 1, "anomalous", "normal")})
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df, x="conflict", hue="label", stat="density", element="step", common_norm=False, bins=40, ax=ax)
    ax.set_xlabel("Conflict coefficient K")
    ax.set_ylabel("Density")
    fig.tight_layout()
    out_path = out_dir / "conflict_distribution.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_threshold_bars(thr_df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    thr_df.plot(x="threshold", y=["recall", "false_alarm"], kind="bar", ax=ax)
    ax.set_ylabel("Rate")
    ax.set_title("Threshold sweep (fused)")
    fig.tight_layout()
    out_path = out_dir / "threshold_sweep.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_forecast_example(preds: dict, df_meta: pd.DataFrame, out_dir: Path) -> Path | None:
    """
    Plot a single forecast vs target curve (first test sample).
    Handles 1D or 2D forecast arrays.
    """
    if "forecast" not in preds or preds["forecast"].size == 0:
        return None
    forecast = preds["forecast"][0]
    future = preds.get("future")
    if forecast.ndim > 1:
        forecast = np.squeeze(forecast)
    if future is not None:
        future = np.squeeze(future[0])
    label = "?"
    if len(df_meta) > 0:
        label = df_meta.reset_index(drop=True).iloc[0].get("h24", "?")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(forecast, label="Forecast h24")
    if future is not None and future.shape == forecast.shape:
        ax.plot(future, label="Future h24 (target)")
    ax.set_title(f"Forecast example | label={label}")
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / "forecast_example.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
# --------------------------------------------------------------------------- #


def generate_report(
    seq_len: int,
    optimize_for: str,
    data_path: Path,
    out_dir: Path,
    results_dir: Path,
    run_suffix: str = "",
    seed: int = 42,
    local_artifacts_path: Path | None = None,
    hub_model_name: str | None = None,
):
    set_seed(seed)
    run_id = f"seq{seq_len}_{optimize_for}{run_suffix}"
    artifacts = ensure_artifacts(
        run_id=run_id,
        out_dir=out_dir,
        results_dir=results_dir,
        local_artifacts_path=local_artifacts_path,
        hub_model_name=hub_model_name,
    )

    cfg = get_best_config(seq_len, optimize_for)
    df, tri_dataset, metadata, splits, loaders, feature_cols = build_loaders(data_path, seq_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    short_cols, long_cols = [0, 1], [2, 3]

    model = WetNet(
        input_dim=len(feature_cols),
        seq_len=seq_len,
        forecast_horizon=24,
        short_count=len(short_cols),
        long_count=len(long_cols),
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(torch.load(artifacts["model"], map_location=device))
    model.eval()

    vib_cfg = json.loads(Path(artifacts["config"]).read_text()).get("vib_config", {})
    vib_model = VIBTransformer(
        input_dim=len(feature_cols),
        seq_len=seq_len,
        d_model=vib_cfg.get("d_model", 128),
        d_content=vib_cfg.get("d_content", 64),
        d_style=vib_cfg.get("d_style", 24),
        nhead=vib_cfg.get("nhead", 4),
        layers=vib_cfg.get("layers", 3),
    ).to(device)
    vib_model.load_state_dict(torch.load(artifacts["vib"], map_location=device))
    vib_model.eval()

    preds = collect_predictions(model, loaders["test"], device)
    metrics = evaluate_multi_horizon(
        torch.from_numpy(preds["probabilities"]),
        torch.from_numpy(preds["targets"]),
        [f"h{h}" for h in HORIZONS],
    )
    metrics_df = pd.DataFrame(list(metrics.items()), columns=["metric", "value"])

    recon_mean = preds["recon_error"].mean()
    recon_std = preds["recon_error"].std() + 1e-6
    fused_prob, conflict_scores = fuse_probabilities(model, vib_model, loaders["test"], recon_mean, recon_std, device)

    labels = metadata.loc[splits["test"], "h24"].to_numpy()
    fusion_rows = sweep_fusion_thresholds(fused_prob, labels, METRIC_THRESHOLDS)
    fusion_rows.append({"metric": "conflict_mean", "value": float(conflict_scores.mean())})
    fusion_df = pd.concat([metrics_df, pd.DataFrame(fusion_rows)], ignore_index=True)

    thr_df = threshold_curves(fused_prob, labels, np.linspace(0.1, 0.9, 9))

    # Save tables
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    fusion_df.to_csv(out_dir / "augmented_metrics.csv", index=False)
    thr_df.to_csv(out_dir / "threshold_sweep.csv", index=False)

    # Plots
    conflict_path = plot_conflict(conflict_scores, labels, out_dir)
    sweep_path = plot_threshold_bars(thr_df, out_dir)
    preds_frame = build_prediction_frame(metadata, splits["test"], preds)
    preds_frame.to_csv(out_dir / "predictions.csv", index=False)
    forecast_path = plot_forecast_example(preds, metadata.iloc[splits["test"]], out_dir)

    # Markdown report
    model_source = (
        f"Local: {local_artifacts_path}"
        if local_artifacts_path
        else (f"Hub: {hub_model_name}" if hub_model_name else "Hub: WetNet/wet-net (default)")
    )
    report = [
        f"# WetNet Evaluation Report ({run_id})",
        "",
        f"- Model source: {model_source}",
        f"- Sequence length: {seq_len}",
        f"- Optimize for: {optimize_for}",
        f"- Test samples: {len(splits['test'])}",
        "",
        "## Key Metrics",
        fusion_df.to_string(index=False),
        "",
        "## Artifacts",
        f"- Conflict histogram: {conflict_path}",
        f"- Threshold sweep: {sweep_path}",
        f"- Forecast example: {forecast_path or 'n/a'}",
        f"- Metrics CSV: {out_dir / 'metrics.csv'}",
        f"- Augmented metrics CSV: {out_dir / 'augmented_metrics.csv'}",
        f"- Threshold sweep CSV: {out_dir / 'threshold_sweep.csv'}",
        f"- Predictions CSV: {out_dir / 'predictions.csv'}",
    ]
    (out_dir / "report.md").write_text("\n".join(report))
    typer.secho(f"Report written to {out_dir / 'report.md'}", fg=typer.colors.GREEN)
    return {
        "metrics": out_dir / "metrics.csv",
        "augmented_metrics": out_dir / "augmented_metrics.csv",
        "threshold_sweep": out_dir / "threshold_sweep.csv",
        "conflict_plot": conflict_path,
        "sweep_plot": sweep_path,
        "forecast_plot": forecast_path,
        "report": out_dir / "report.md",
    }


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


@app.command()
def evaluate(
    seq_len: int = typer.Option(96, help="Sequence length used during training."),
    optimize_for: str = typer.Option("recall", help="recall or false_alarm; matches saved config."),
    data_path: str | None = typer.Option(None, help="Preprocessed parquet path; defaults to processed output."),
    output_dir: str = typer.Option("./results/wetnet/report", help="Where to write report/plots."),
    results_dir: str = typer.Option("./results", help="Directory where training results are stored."),
    run_suffix: str = typer.Option("", help="Suffix used during training (e.g., _fast) to locate artifacts."),
    seed: int = typer.Option(42, help="Random seed."),
    local_artifacts_path: str | None = typer.Option(
        None,
        "--local-artifacts-path",
        help="Path to local directory containing model artifacts (wetnet.pt, vib.pt, config.json).",
    ),
    hub_model_name: str | None = typer.Option(
        None,
        "--hub-model-name",
        help=(
            "Hugging Face repo to pull artifacts from. "
            "Defaults to WetNet/wet-net if neither --local-artifacts-path nor --hub-model-name is specified."
        ),
    ),
):
    """
    Run full evaluation + reporting without retraining.

    Model source priority:
    1. --local-artifacts-path (if provided, must exist)
    2. --hub-model-name (if provided, downloads from HuggingFace Hub)
    3. Default: checks local training directory, then falls back to WetNet/wet-net
    """
    if seq_len not in SEQ_LENGTHS:
        typer.secho(f"seq_len must be one of {SEQ_LENGTHS}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Validate that only one of local_artifacts_path or hub_model_name is provided
    if local_artifacts_path is not None and hub_model_name is not None:
        typer.secho(
            "Cannot specify both --local-artifacts-path and --hub-model-name. Choose one.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    # If data_path is explicitly provided, it must exist (no fallback).
    if data_path:
        data_path_obj = Path(data_path)
        if not data_path_obj.exists():
            typer.secho(
                f"Preprocessed parquet not found at {data_path_obj}. Run `wet-net pre-process` first.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        final_data_path = data_path_obj
    else:
        # Fallback to default processed parquet location
        from wet_net.data.preprocess import PROCESSED_PARQUET

        final_data_path = PROCESSED_PARQUET
        if not final_data_path.exists():
            typer.secho(
                f"Preprocessed parquet not found at {final_data_path}. Run `wet-net pre-process` first.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

    local_artifacts_path_obj = Path(local_artifacts_path) if local_artifacts_path else None
    results_dir_path = Path(results_dir)
    # If output_dir is the default, construct it from results_dir to respect --results-dir
    if output_dir == "./results/wetnet/report":
        output_dir = str(results_dir_path / "wetnet" / "report")
    out_dir = Path(output_dir) / f"seq{seq_len}_{optimize_for}{run_suffix}"
    try:
        generate_report(
            seq_len=seq_len,
            optimize_for=optimize_for,
            data_path=final_data_path,
            out_dir=out_dir,
            results_dir=results_dir_path,
            run_suffix=run_suffix,
            seed=seed,
            local_artifacts_path=local_artifacts_path_obj,
            hub_model_name=hub_model_name,
        )
    except FileNotFoundError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(code=1) from e
    except RepositoryNotFoundError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
