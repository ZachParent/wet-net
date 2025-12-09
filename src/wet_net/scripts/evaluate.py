import json
from pathlib import Path

import typer
import torch
import pandas as pd

from wet_net.config.tri_task import SEQ_LENGTHS, get_best_config, HORIZONS, METRIC_THRESHOLDS
from wet_net.data.datasets import (
    build_metadata,
    build_policy_split,
    compute_class_weights,
    compute_future_sequences,
    ensure_anomaly_coverage,
    make_dataloaders,
    TimeSeriesDataset,
    TriTaskWindowDataset,
)
from wet_net.data.preprocess import load_preprocessed_dataframe, select_feature_columns
from wet_net.eval.metrics import evaluate_multi_horizon, sweep_fusion_thresholds
from wet_net.eval.predictions import build_prediction_frame, collect_predictions
from wet_net.models.vib import VIBTransformer
from wet_net.models.wetnet import WetNet
from wet_net.paths import RESULTS_DIR
from wet_net.training.fusion import fuse_probabilities
from wet_net.training.utils import batch_for_seq, intelligent_batch_size, max_samples_for_seq, stride_for_seq

app = typer.Typer()


def build_loaders(seq_len: int, preprocessed: Path, feature_cols):
    df = load_preprocessed_dataframe(preprocessed)
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
    return df, tri_dataset, metadata, splits, loaders


@app.command()
def evaluate(
    seq_len: int = typer.Option(96, help="Sequence length used during training."),
    optimize_for: str = typer.Option("recall", help="recall or false_alarm; matches saved config."),
    mock: bool = typer.Option(False, "--mock", help="Use mock dataset."),
    artifacts_dir: str | None = typer.Option(None, help="Path to saved artifacts (defaults to results/wetnet/seq*_*)"),
):
    """
    Re-evaluate saved models (no retraining).
    """
    if seq_len not in SEQ_LENGTHS:
        typer.secho(f"seq_len must be one of {SEQ_LENGTHS}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    cfg = get_best_config(seq_len, optimize_for)
    run_id = f"seq{seq_len}_{optimize_for}"
    art_dir = Path(artifacts_dir) if artifacts_dir else RESULTS_DIR / "wetnet" / run_id
    model_path = art_dir / "wetnet.pt"
    vib_path = art_dir / "vib.pt"
    config_path = art_dir / "config.json"
    if not model_path.exists() or not vib_path.exists():
        typer.secho(f"Artifacts not found in {art_dir}. Run `wet-net train` first.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    preprocessed = Path("data/processed/mock_preprocessed.parquet" if mock else "data/processed/anomalous_consumption_preprocessed.parquet")
    if not preprocessed.exists():
        if mock:
            typer.secho("Mock data missing. Run `wet-net pre-process --mock` first.", fg=typer.colors.RED)
        else:
            typer.secho("Preprocessed real data missing. Run `wet-net pre-process --data-url <url>` first.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, tri_dataset, metadata, splits, loaders = build_loaders(seq_len, preprocessed, select_feature_columns(load_preprocessed_dataframe(preprocessed)))

    feature_cols = select_feature_columns(df)
    short_cols = [0, 1]
    long_cols = [2, 3]
    pos_weight_short = compute_class_weights(tri_dataset.base.targets, short_cols)
    pos_weight_long = compute_class_weights(tri_dataset.base.targets, long_cols)

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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    vib_cfg = json.loads(config_path.read_text()).get("vib_config", {})
    vib_model = VIBTransformer(
        input_dim=len(feature_cols),
        seq_len=seq_len,
        d_model=vib_cfg.get("d_model", 128),
        d_content=vib_cfg.get("d_content", 64),
        d_style=vib_cfg.get("d_style", 24),
        nhead=vib_cfg.get("nhead", 4),
        layers=vib_cfg.get("layers", 3),
    ).to(device)
    vib_model.load_state_dict(torch.load(vib_path, map_location=device))
    vib_model.eval()

    preds = collect_predictions(model, loaders["test"], device)
    metrics = evaluate_multi_horizon(
        torch.from_numpy(preds["probabilities"]), torch.from_numpy(preds["targets"]), [f"h{h}" for h in HORIZONS]
    )
    metrics_df = pd.DataFrame(list(metrics.items()), columns=["metric", "value"])

    recon_mean = preds["recon_error"].mean()
    recon_std = preds["recon_error"].std() + 1e-6
    fused_prob, conflict_scores = fuse_probabilities(model, vib_model, loaders["test"], recon_mean, recon_std, device)
    labels = metadata.loc[splits["test"], "h24"].to_numpy()
    fusion_rows = sweep_fusion_thresholds(fused_prob, labels, METRIC_THRESHOLDS)
    fusion_rows.append({"metric": "conflict_mean", "value": float(conflict_scores.mean())})
    fusion_df = pd.concat([metrics_df, pd.DataFrame(fusion_rows)], ignore_index=True)
    out_path = art_dir / "re_evaluated_metrics.csv"
    fusion_df.to_csv(out_path, index=False)
    typer.secho(f"Saved metrics to {out_path}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
