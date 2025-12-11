from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer

from wet_net.data.preprocess import PROCESSED_PARQUET
from wet_net.pipelines.tri_task import build_data_bundle

app = typer.Typer(help="Lightweight dataset visualizations (e.g., class balance).")


def _resolve_preprocessed_path(data_path: str | None, mock: bool) -> Path:
    if data_path:
        return Path(data_path)
    if mock:
        return Path("./data/processed/mock_preprocessed.parquet")
    return PROCESSED_PARQUET


def _attach_split_labels(meta: pd.DataFrame, splits: dict[str, list[int]]) -> pd.DataFrame:
    meta = meta.set_index("dataset_idx")
    meta["split"] = "unassigned"
    for split, idxs in splits.items():
        if len(idxs) > 0:
            meta.loc[idxs, "split"] = split
    return meta.reset_index()


def _plot_overall_balance(meta: pd.DataFrame, out_path: Path) -> Path:
    counts = meta["any_anomaly"].value_counts().rename(index={0.0: "normal", 1.0: "anomalous"}).reset_index()
    counts.columns = ["label", "count"]
    labels = counts["label"].tolist()
    palette = ["#4c9aff", "#ff6b6b"][: len(labels)]
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.barplot(data=counts, x="label", y="count", hue="label", palette=palette, legend=False, ax=ax)
    ax.set_ylabel("Samples")
    ax.set_xlabel("")
    ax.set_title("Overall class balance")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _plot_split_balance(meta: pd.DataFrame, out_path: Path) -> Path:
    df = meta.copy()
    df["label"] = df["any_anomaly"].map({0.0: "normal", 1.0: "anomalous"})
    split_counts = df.groupby(["split", "label"]).size().reset_index(name="count")
    labels = split_counts["label"].unique().tolist()
    palette = ["#4c9aff", "#ff6b6b"][: len(labels)]
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=split_counts, x="split", y="count", hue="label", palette=palette, ax=ax)
    ax.set_ylabel("Samples")
    ax.set_xlabel("")
    ax.set_title("Class balance by split")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


@app.command("class-balance")
def class_balance(
    seq_len: int = typer.Option(96, help="Sequence length (matches training/eval)."),
    mock: bool = typer.Option(False, help="Use mock dataset paths and allow regeneration if too short."),
    data_path: str | None = typer.Option(None, help="Path to preprocessed parquet; defaults to canonical location."),
    out_dir: str | None = typer.Option(None, help="Directory to store plots (defaults to results/plots)."),
):
    """
    Plot class balance (overall and per split) using the existing preprocessing and split logic.
    """
    preprocessed = _resolve_preprocessed_path(data_path, mock)
    if not preprocessed.exists():
        typer.secho(
            f"Preprocessed parquet not found at {preprocessed}. Run `wet-net pre-process` first.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    bundle = build_data_bundle(preprocessed, seq_len, allow_mock_regen=mock)
    meta = _attach_split_labels(bundle.metadata, bundle.splits)

    out_base = Path(out_dir) if out_dir else Path("./results/plots")
    overall_path = out_base / f"class_balance_overall_seq{seq_len}.png"
    split_path = out_base / f"class_balance_splits_seq{seq_len}.png"

    _plot_overall_balance(meta, overall_path)
    _plot_split_balance(meta, split_path)

    total_anom = int(meta["any_anomaly"].sum())
    total = len(meta)
    ratio = (total_anom / total * 100) if total else 0.0
    typer.secho(
        f"Saved class-balance plots to {out_base} | anomalies={total_anom}/{total} ({ratio:.2f}%).",
        fg=typer.colors.GREEN,
    )
    typer.secho(f"Overall: {overall_path.name}", fg=typer.colors.BLUE)
    typer.secho(f"By split: {split_path.name}", fg=typer.colors.BLUE)


if __name__ == "__main__":
    app()
