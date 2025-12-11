"""
Evaluation helpers.
"""

from __future__ import annotations

import numpy as np
import torch


def evaluate_multi_horizon(
    probabilities: torch.Tensor, targets: torch.Tensor, horizon_names: list | None = None
) -> dict:
    stats = {}
    if horizon_names is None:
        horizon_names = [f"h{i}" for i in range(probabilities.shape[1])]
    for i, name in enumerate(horizon_names):
        probs = probabilities[:, i]
        t = targets[:, i]
        total_anomalies = t.sum().item()
        total_samples = len(t)
        if total_anomalies == 0:
            stats[f"anomalies_found_{name}"] = 0
            stats[f"anomalies_missed_{name}"] = 0
            stats[f"anomaly_detection_rate_{name}"] = 0.0
            stats[f"total_anomalies_{name}"] = 0
            stats[f"false_positives_{name}"] = 0
            continue
        threshold = 0.5
        bin_p = (probs >= threshold).float()
        tp = ((bin_p == 1) & (t == 1)).sum().item()
        fn = ((bin_p == 0) & (t == 1)).sum().item()
        fp = ((bin_p == 1) & (t == 0)).sum().item()
        stats[f"anomalies_found_{name}"] = tp
        stats[f"anomalies_missed_{name}"] = fn
        stats[f"anomaly_detection_rate_{name}"] = tp / total_anomalies if total_anomalies > 0 else 0.0
        stats[f"total_anomalies_{name}"] = total_anomalies
        stats[f"false_positives_{name}"] = fp
        stats[f"total_samples_{name}"] = total_samples
        stats[f"precision_{name}"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        stats[f"f1_{name}"] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return stats


def sweep_fusion_thresholds(fused_prob: np.ndarray, labels: np.ndarray, thresholds: list[float]) -> list[dict]:
    rows = []
    for thr in thresholds:
        alert_mask = fused_prob >= thr
        fused_recall = float(alert_mask[labels == 1].mean()) if np.any(labels == 1) else float("nan")
        fused_false_alarm = float(alert_mask[labels == 0].mean()) if np.any(labels == 0) else float("nan")
        rows.append({"metric": f"fused_recall@{thr:.1f}", "value": fused_recall})
        rows.append({"metric": f"fused_false_alarm@{thr:.1f}", "value": fused_false_alarm})
    return rows
