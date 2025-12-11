from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from wet_net.config.tri_task import (
    FORECAST_HORIZON,
    HORIZONS,
    METRIC_THRESHOLDS,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    get_best_config,
)
from wet_net.data.datasets import (
    DataBundle,
    TimeSeriesDataset,
    TriTaskWindowDataset,
    build_metadata,
    build_policy_split,
    compute_class_weights,
    compute_future_sequences,
    ensure_anomaly_coverage,
    make_balanced_loader,
    make_dataloaders,
)
from wet_net.data.preprocess import load_preprocessed_dataframe, select_feature_columns
from wet_net.eval.metrics import evaluate_multi_horizon, sweep_fusion_thresholds
from wet_net.eval.predictions import build_prediction_frame, collect_predictions
from wet_net.models.vib import VIBTransformer
from wet_net.models.wetnet import WetNet
from wet_net.training.fusion import fuse_probabilities
from wet_net.training.loops import build_training_stages, train_staged_model
from wet_net.training.utils import (
    batch_for_seq,
    intelligent_batch_size,
    max_samples_for_seq,
    set_seed,
    stride_for_seq,
)

console = Console()


def build_data_bundle(
    preprocessed_path: Path,
    seq_len: int,
    allow_mock_regen: bool = False,
    min_anomaly_ratio: float = 0.0,
) -> DataBundle:
    df = load_preprocessed_dataframe(preprocessed_path)
    # If mock data is too short, regenerate a larger mock dataset
    if allow_mock_regen and len(df) < seq_len + FORECAST_HORIZON + 1:
        from wet_net.data.preprocess import prepare_dataset

        preprocessed_path = prepare_dataset(mock=True, force_mock_regen=True)
        df = load_preprocessed_dataframe(preprocessed_path)
    feature_cols = select_feature_columns(df)
    base_batch = batch_for_seq(seq_len)
    stride = stride_for_seq(seq_len)
    max_samples = max_samples_for_seq(seq_len)
    env_cap = os.getenv("WETNET_MAX_SAMPLES")
    if env_cap:
        try:
            cap = int(env_cap)
            max_samples = min(max_samples, cap)
        except ValueError:
            pass
    base_dataset = TimeSeriesDataset(
        df,
        seq_len=seq_len,
        horizons=HORIZONS,
        stride=stride,
        max_samples=max_samples,
        feature_cols=feature_cols,
    )
    future_targets, anchor_times, policy_marks = compute_future_sequences(df, base_dataset, FORECAST_HORIZON)
    tri_dataset = TriTaskWindowDataset(base_dataset, future_targets)
    metadata = build_metadata(base_dataset, anchor_times, policy_marks, HORIZONS)
    splits = build_policy_split(metadata, (TRAIN_RATIO, VAL_RATIO, TEST_RATIO))
    ensure_anomaly_coverage(metadata, splits, min_ratio=min_anomaly_ratio)
    batch_size = intelligent_batch_size(seq_len, len(feature_cols), base_batch, d_model_guess=256)
    loaders = make_dataloaders(tri_dataset, splits, batch_size)
    return DataBundle(df=df, dataset=tri_dataset, metadata=metadata, splits=splits, loaders=loaders)


def train_wetnet(
    seq_len: int,
    optimize_for: str,
    preprocessed_path: Path,
    device: torch.device,
    results_dir: Path,
    vib_cfg_overrides: dict | None = None,
    mock: bool = False,
    seed: int | None = None,
    fast_epochs: int | None = None,
    run_suffix: str = "",
    early_stop: bool = True,
    min_delta_abs: float = 1e-4,
    min_delta_rel: float = 0.0,
    min_anomaly_ratio: float = 0.0,
    task_weights: dict[str, float] | None = None,
    monitor: str = "total",
) -> dict[str, Path]:
    set_seed(seed)
    cfg = get_best_config(seq_len, optimize_for)
    bundle = build_data_bundle(preprocessed_path, seq_len, allow_mock_regen=mock, min_anomaly_ratio=min_anomaly_ratio)
    feature_cols = select_feature_columns(bundle.df)
    short_cols = [0, 1]
    long_cols = [2, 3]
    pos_weight_short = compute_class_weights(bundle.dataset.base.targets, short_cols)
    pos_weight_long = compute_class_weights(bundle.dataset.base.targets, long_cols)

    model = WetNet(
        input_dim=len(feature_cols),
        seq_len=seq_len,
        forecast_horizon=FORECAST_HORIZON,
        short_count=len(short_cols),
        long_count=len(long_cols),
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
    ).to(device)

    stages = build_training_stages(cfg.schedule_variant, cfg.pcgrad)

    # Apply fast-mode overrides (env or CLI)
    fast_cap = fast_epochs
    fast_env = os.getenv("WETNET_FAST")
    if fast_cap is None and fast_env:
        try:
            fast_cap = int(fast_env)
        except ValueError:
            fast_cap = 2
    if fast_cap is not None:
        for st in stages:
            st["epochs"] = min(st["epochs"], fast_cap)
            st["patience"] = min(st.get("patience", st["epochs"]), max(1, fast_cap // 2))
            st["min_epochs"] = max(1, min(st.get("min_epochs", st["epochs"]), fast_cap // 2))
    if mock:
        for st in stages:
            st["epochs"] = min(2, st["epochs"])
            st["patience"] = 1
    total_epochs = sum(s["epochs"] for s in stages)
    split_summary = None
    try:
        from wet_net.data.datasets import summarize_splits

        split_summary = summarize_splits(bundle.metadata, bundle.splits)
    except ImportError:
        # Optional summarize_splits function not available
        split_summary = None
    console.print(
        f"[bold cyan]Training plan[/bold cyan] seq_len={seq_len}, optimize_for={optimize_for} "
        f"(schedule={cfg.schedule_variant}, pcgrad={'on' if cfg.pcgrad else 'off'})"
    )
    console.print(f"Total stages={len(stages)}, total epochs={total_epochs}")
    if split_summary is not None:
        console.print("[bold]Split anomaly ratios[/bold]")
        for _, row in split_summary.iterrows():
            console.print(
                f"  {row['split']}: samples={row['samples']}, policies={row['policies']}, "
                f"anomaly_ratio={row['anomaly_ratio']:.4f}"
            )
    for idx, st in enumerate(stages, 1):
        patience = st.get("patience", st["epochs"])
        console.print(
            f"  {idx}. {st['name']}: {st['epochs']} ep, lr={st['lr']}, "
            f"patience={patience}, pcgrad={'on' if st['pcgrad'] else 'off'}"
        )
    history = train_staged_model(
        model,
        bundle.loaders,
        stages,
        pos_weight_short,
        pos_weight_long,
        device,
        early_stop=early_stop,
        min_delta_abs=min_delta_abs,
        min_delta_rel=min_delta_rel,
        task_weights=task_weights,
        monitor=monitor,
    )

    # Summarise how many epochs actually ran per stage (helps spot early stopping)
    epochs_ran: dict[str, int] = {}
    for row in history:
        stage = row["stage"]
        epochs_ran[stage] = max(epochs_ran.get(stage, -1), row["epoch"])
    if epochs_ran:
        console.print("[bold green]Stage completion summary[/bold green]")
        for st in stages:
            planned = st["epochs"]
            ran = epochs_ran.get(st["name"], -1) + 1
            note = "" if early_stop else " (early-stop disabled)"
            if ran < planned and early_stop:
                note = f" (stopped early at {ran}/{planned})"
            console.print(f"  {st['name']}: {ran} / {planned}{note}")

    vib_base = {
        "seq_len": seq_len,
        "d_model": 128,
        "d_content": 64,
        "d_style": 24,
        "nhead": 4,
        "layers": 3,
        "epochs": 40,
        "steps_per_epoch": 80,
        "cls_weight": 2.0,
        "beta": 7.5e-4,
    }
    if vib_cfg_overrides:
        vib_base.update(vib_cfg_overrides)
    if fast_cap is not None:
        vib_base["epochs"] = min(vib_base["epochs"], max(2, fast_cap))
        vib_base["steps_per_epoch"] = min(vib_base["steps_per_epoch"], max(5, fast_cap * 5))
    if mock:
        vib_base["epochs"] = min(5, vib_base["epochs"])
        vib_base["steps_per_epoch"] = min(10, vib_base["steps_per_epoch"])

    console.print(
        f"[bold cyan]VIB training[/bold cyan] epochs={vib_base['epochs']}, "
        f"steps/epoch={vib_base['steps_per_epoch']}, beta={vib_base['beta']}"
    )
    vib_model = VIBTransformer(
        input_dim=len(feature_cols),
        seq_len=seq_len,
        d_model=vib_base["d_model"],
        d_content=vib_base["d_content"],
        d_style=vib_base["d_style"],
        nhead=vib_base["nhead"],
        layers=vib_base["layers"],
    ).to(device)

    # quick probe training (keep lightweight; mirrors notebook defaults)
    balanced_loader = make_balanced_loader(
        bundle.dataset, bundle.splits["train"], batch_size=bundle.loaders["train"].batch_size
    )
    optimizer = torch.optim.AdamW(vib_model.parameters(), lr=1e-3)
    iterator = iter(balanced_loader)
    vib_model.train()
    with Progress(
        TextColumn("[magenta]VIB {task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} ep"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as vib_progress:
        vib_task = vib_progress.add_task("training", total=vib_base["epochs"])
        for epoch in range(vib_base["epochs"]):
            total = 0.0
            for _ in range(vib_base["steps_per_epoch"]):
                try:
                    seq, targets, _ = next(iterator)
                except StopIteration:
                    iterator = iter(balanced_loader)
                    seq, targets, _ = next(iterator)
                seq = seq.to(device)
                labels = targets[:, 0:1].to(device)
                recon, logits, c_mu, c_logvar, s_mu, s_logvar = vib_model(seq)
                loss_rec = torch.nn.functional.mse_loss(recon, seq)
                loss_cls = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                kl_c = -0.5 * torch.mean(1 + c_logvar - c_mu.pow(2) - c_logvar.exp())
                kl_s = -0.5 * torch.mean(1 + s_logvar - s_mu.pow(2) - s_logvar.exp())
                loss = loss_rec + vib_base["cls_weight"] * loss_cls + vib_base["beta"] * (kl_c + kl_s)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
            if epoch % 5 == 0 or epoch == vib_base["epochs"] - 1:
                avg_loss = total / vib_base["steps_per_epoch"]
                console.log(f"VIB epoch {epoch + 1}/{vib_base['epochs']} | loss={avg_loss:.4f}")
            vib_progress.advance(vib_task, 1)

    artifacts: dict[str, Path] = {}
    run_id = f"seq{seq_len}_{optimize_for}{run_suffix}"
    out_dir = results_dir / "wetnet" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "wetnet.pt"
    vib_path = out_dir / "vib.pt"
    hist_path = out_dir / "history.json"
    torch.save(model.state_dict(), model_path)
    torch.save(vib_model.state_dict(), vib_path)
    hist_path.write_text(json.dumps(history, indent=2))
    artifacts.update({"model": model_path, "vib": vib_path, "history": hist_path})

    # Evaluation
    preds = collect_predictions(model, bundle.loaders["test"], device)
    metrics = evaluate_multi_horizon(
        torch.from_numpy(preds["probabilities"]), torch.from_numpy(preds["targets"]), [f"h{h}" for h in HORIZONS]
    )
    metrics_df = pd.DataFrame(list(metrics.items()), columns=["metric", "value"])
    metrics_path = out_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    artifacts["metrics"] = metrics_path

    recon_mean = preds["recon_error"].mean()
    recon_std = preds["recon_error"].std() + 1e-6
    vib_model.eval()
    fused_prob, conflict_scores = fuse_probabilities(
        model, vib_model, bundle.loaders["test"], recon_mean, recon_std, device
    )
    labels = bundle.metadata.loc[bundle.splits["test"], "h24"].to_numpy()
    fusion_rows = sweep_fusion_thresholds(fused_prob, labels, METRIC_THRESHOLDS)
    fusion_rows.append({"metric": "conflict_mean", "value": float(conflict_scores.mean())})
    fusion_df = pd.concat([metrics_df, pd.DataFrame(fusion_rows)], ignore_index=True)
    fusion_path = out_dir / "augmented_metrics.csv"
    fusion_df.to_csv(fusion_path, index=False)
    # extract the requested threshold summary for quick lookup
    target_thr = cfg.threshold
    thr_recall = fusion_df.loc[fusion_df["metric"] == f"fused_recall@{target_thr:.1f}", "value"].max()
    thr_fa = fusion_df.loc[fusion_df["metric"] == f"fused_false_alarm@{target_thr:.1f}", "value"].max()
    summary = {"threshold": target_thr, "recall": float(thr_recall), "false_alarm": float(thr_fa)}
    artifacts["augmented_metrics"] = fusion_path

    preds_frame = build_prediction_frame(bundle.metadata, bundle.splits["test"], preds)
    preds_path = out_dir / "predictions.csv"
    preds_frame.to_csv(preds_path, index=False)
    artifacts["predictions"] = preds_path

    cfg_path = out_dir / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "optimize_for": optimize_for,
                "seq_len": seq_len,
                "model_config": cfg.__dict__,
                "vib_config": vib_base,
                "threshold_summary": summary,
            },
            indent=2,
        )
    )
    artifacts["config"] = cfg_path
    return artifacts
