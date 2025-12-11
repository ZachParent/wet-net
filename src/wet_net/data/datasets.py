"""
Dataset helpers for WetNet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from wet_net.data.preprocess import FEATURE_COLS


class TimeSeriesDataset(Dataset):
    """
    Multi-horizon time series dataset for anomaly detection.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        seq_len: int,
        horizons: list[int] | None = None,
        stride: int = 1,
        max_samples: int | None = None,
        feature_cols: list[str] | None = None,
    ):
        self.seq_len = seq_len
        self.horizons = horizons or [24, 48, 168, 336]
        self.max_horizon = max(self.horizons)
        feature_cols = feature_cols or FEATURE_COLS

        existing_cols = [c for c in feature_cols if c in data.columns]
        self.features = data[existing_cols].to_numpy().astype(np.float32)
        self.n_features = len(existing_cols)

        self.labels = data["is_anomalous"].to_numpy().astype(np.float32)
        self.ids = data["POLISSA_SUBM"].astype("category").cat.codes.to_numpy()
        self.cum_labels = data.groupby("POLISSA_SUBM")["is_anomalous"].cumsum().to_numpy().astype(np.float32)

        full_indices = self._compute_indices(stride)
        if len(full_indices) == 0:
            self.indices = np.empty(0, dtype=np.int32)
            self.targets = np.empty((0, len(self.horizons)), dtype=np.float32)
            self.pos_ratio = 0.0
            return

        self.targets = self._compute_multi_targets(full_indices)
        self.indices = full_indices

        if max_samples and len(self.indices) > max_samples:
            self._stratified_subsample(max_samples)

        self.pos_ratio = float(self.targets[:, 0].mean()) if len(self.targets) > 0 else 0.0

    def _compute_indices(self, stride: int) -> np.ndarray:
        boundaries = np.where(self.ids[:-1] != self.ids[1:])[0] + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(self.features)]))

        indices = []
        full_window_req = self.seq_len + self.max_horizon
        for s, e in zip(starts, ends, strict=True):
            if (e - s) <= full_window_req:
                continue
            idx = np.arange(s, e - full_window_req + 1, step=stride)
            indices.extend(idx)
        return np.array(indices, dtype=np.int32)

    def _compute_multi_targets(self, indices: np.ndarray) -> np.ndarray:
        targets = []
        pred_start_indices = indices + self.seq_len
        base_cum = self.cum_labels[pred_start_indices - 1]
        for h in self.horizons:
            pred_end_indices = pred_start_indices + h - 1
            sums = self.cum_labels[pred_end_indices] - base_cum
            targets.append((sums > 0).astype(np.float32))
        return np.stack(targets, axis=1)

    def _stratified_subsample(self, max_samples: int):
        rng = np.random.default_rng(42)
        any_anomaly = self.targets.max(axis=1)
        pos_idx = np.where(any_anomaly == 1.0)[0]
        neg_idx = np.where(any_anomaly == 0.0)[0]

        total = len(self.indices)
        if total <= max_samples:
            return

        if len(pos_idx) == 0 or len(neg_idx) == 0:
            chosen = rng.choice(total, max_samples, replace=False)
        else:
            # Preserve the original class ratio while guaranteeing both classes appear.
            pos_frac = len(pos_idx) / total
            pos_take = int(round(max_samples * pos_frac))
            pos_take = max(1, min(len(pos_idx), pos_take))
            neg_take = max_samples - pos_take
            neg_take = max(1, min(len(neg_idx), neg_take))

            # If the rounding/clamping pushed us over budget, trim the larger class.
            overflow = pos_take + neg_take - max_samples
            if overflow > 0:
                if pos_take > neg_take:
                    pos_take = max(1, pos_take - overflow)
                else:
                    neg_take = max(1, neg_take - overflow)

            # If we are under budget (happens when one class is scarce), fill from the class with headroom.
            deficit = max_samples - (pos_take + neg_take)
            if deficit > 0:
                pos_room = len(pos_idx) - pos_take
                neg_room = len(neg_idx) - neg_take
                if pos_room >= neg_room:
                    add = min(pos_room, deficit)
                    pos_take += add
                    deficit -= add
                if deficit > 0 and neg_room > 0:
                    add = min(neg_room, deficit)
                    neg_take += add

            chosen_pos = rng.choice(pos_idx, pos_take, replace=False) if pos_take > 0 else np.array([], dtype=int)
            chosen_neg = rng.choice(neg_idx, neg_take, replace=False) if neg_take > 0 else np.array([], dtype=int)
            chosen = np.concatenate([chosen_pos, chosen_neg])
        rng.shuffle(chosen)
        self.indices = self.indices[chosen]
        self.targets = self.targets[chosen]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        x = torch.from_numpy(self.features[t : t + self.seq_len])
        y = torch.from_numpy(self.targets[idx])
        return x, y


class TriTaskWindowDataset(Dataset):
    """Wrap TimeSeriesDataset so each sample also carries future 24h sequences."""

    def __init__(self, base_dataset: TimeSeriesDataset, future_targets: np.ndarray):
        if len(base_dataset) != len(future_targets):
            raise ValueError("Future targets must align with dataset windows.")
        self.base = base_dataset
        self.future_targets = torch.from_numpy(future_targets.astype(np.float32))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        seq, horizons = self.base[idx]
        future = self.future_targets[idx]
        return seq, horizons, future


def compute_future_sequences(
    df: pd.DataFrame,
    dataset: TimeSeriesDataset,
    forecast_horizon: int,
    consumption_col: str = "CONSUMO_REAL_norm",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = df[consumption_col].to_numpy().astype(np.float32)
    timestamps = df["FECHA_HORA"].to_numpy()
    policies = df["POLISSA_SUBM"].to_numpy()
    seq_len = dataset.seq_len
    futures, anchors, policy_marks = [], [], []
    for start in dataset.indices:
        pred_start = start + seq_len
        pred_end = pred_start + forecast_horizon
        future_slice = values[pred_start:pred_end]
        if len(future_slice) != forecast_horizon:
            continue
        futures.append(future_slice)
        anchors.append(timestamps[pred_start - 1])
        policy_marks.append(policies[pred_start - 1])
    return np.stack(futures), np.asarray(anchors), np.asarray(policy_marks)


def build_metadata(
    dataset: TimeSeriesDataset,
    anchors: np.ndarray,
    policies: np.ndarray,
    horizons: list[int],
) -> pd.DataFrame:
    meta = pd.DataFrame(
        {
            "dataset_idx": np.arange(len(dataset)),
            "timestamp": pd.to_datetime(anchors),
            "policy": policies,
            "any_anomaly": dataset.targets.max(axis=1),
        }
    )
    for i, horizon in enumerate(horizons):
        meta[f"h{horizon}"] = dataset.targets[:, i]
    return meta


def build_policy_split(meta_df: pd.DataFrame, ratios: tuple[float, float, float]) -> dict[str, list[int]]:
    train_ratio, val_ratio, test_ratio = ratios
    splits = {"train": [], "val": [], "test": []}
    for _, group in meta_df.sort_values("timestamp").groupby("policy"):
        ordered = group.sort_values("timestamp")["dataset_idx"].to_numpy()
        n = len(ordered)
        if n == 0:
            continue
        train_end = max(1, min(int(np.floor(n * train_ratio)), n - 2)) if n > 2 else max(1, n - 2)
        val_end = (
            max(train_end + 1, min(int(np.floor(n * (train_ratio + val_ratio))), n - 1)) if n > 2 else train_end + 1
        )
        splits["train"].extend(ordered[:train_end].tolist())
        splits["val"].extend(ordered[train_end:val_end].tolist())
        splits["test"].extend(ordered[val_end:].tolist())
    return splits


def ensure_anomaly_coverage(
    meta_df: pd.DataFrame, splits: dict[str, list[int]], min_ratio: float = 0.0, max_transfer: int | None = None
) -> None:
    """
    Guarantee each non-train split has anomalies.
    min_ratio: minimum fraction of samples in each target split that should be anomalous.
    max_transfer: hard cap of how many anomalous samples can be pulled from train overall.
    """
    if meta_df["any_anomaly"].sum() == 0:
        return
    anomaly_pool = [idx for idx in splits["train"] if meta_df.loc[idx, "any_anomaly"] == 1]
    transferred = 0
    for target in ("val", "test"):
        target_idxs = splits[target]
        target_anoms = [i for i in target_idxs if meta_df.loc[i, "any_anomaly"] == 1]
        desired = int(np.ceil(min_ratio * len(target_idxs))) if min_ratio > 0 else 0
        needed = max(0, desired - len(target_anoms))
        if len(target_anoms) == 0:
            needed = max(1, needed)  # at least one anomaly
        if max_transfer is not None:
            remaining = max(0, max_transfer - transferred)
            needed = min(needed, remaining)
        if needed <= 0 or not anomaly_pool:
            continue
        take = min(needed, len(anomaly_pool))
        donate = anomaly_pool[:take]
        anomaly_pool = anomaly_pool[take:]
        transferred += take
        for idx in donate:
            splits["train"].remove(idx)
            splits[target].append(idx)


def summarize_splits(meta_df: pd.DataFrame, splits: dict[str, list[int]]) -> pd.DataFrame:
    rows = []
    for split, idxs in splits.items():
        subset = meta_df.loc[idxs] if idxs else pd.DataFrame(columns=meta_df.columns)
        rows.append(
            {
                "split": split,
                "samples": len(idxs),
                "policies": subset["policy"].nunique() if not subset.empty else 0,
                "anomaly_ratio": subset["any_anomaly"].mean() if not subset.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def make_dataloaders(dataset: Dataset, splits: dict[str, list[int]], batch_size: int) -> dict[str, DataLoader]:
    loaders = {}
    for split, idxs in splits.items():
        subset = Subset(dataset, idxs)
        loaders[split] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            drop_last=(split == "train"),
        )
    return loaders


def make_balanced_loader(dataset: Dataset, indices: list[int], batch_size: int) -> DataLoader:
    if not indices:
        raise ValueError("Cannot build balanced loader without indices.")
    subset = Subset(dataset, indices)
    labels = dataset.base.targets[indices, 0] if hasattr(dataset, "base") else dataset.targets[indices, 0]
    pos = max(1, labels.sum())
    neg = max(1, len(labels) - labels.sum())
    weights = np.where(labels > 0, 0.5 / pos, 0.5 / neg).astype(np.float64)
    sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(indices), replacement=True)
    return DataLoader(subset, batch_size=batch_size, sampler=sampler, drop_last=True)


def compute_class_weights(targets: np.ndarray, cols: list[int]) -> torch.Tensor:
    rates = targets[:, cols].mean(axis=0)
    weights = ((1 - rates) / (rates + 1e-6)).astype(np.float32)
    return torch.from_numpy(weights)


@dataclass
class DataBundle:
    df: pd.DataFrame
    dataset: TriTaskWindowDataset
    metadata: pd.DataFrame
    splits: dict[str, list[int]]
    loaders: dict[str, DataLoader]
