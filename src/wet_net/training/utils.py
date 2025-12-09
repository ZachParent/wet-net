from __future__ import annotations

import random

import numpy as np
import torch

EFFECTIVE_BATCH_SIZE = 32


def stride_for_seq(seq_len: int) -> int:
    return max(1, seq_len // 96)


def batch_for_seq(seq_len: int) -> int:
    if seq_len >= 720:
        return 2
    if seq_len >= 360:
        return 4
    return 8


def max_samples_for_seq(seq_len: int) -> int:
    base = 120_000
    scaled = int(base * 96 / max(96, seq_len))
    return max(8_000, min(150_000, scaled))


def estimate_per_sample_bytes(seq_len: int, feature_count: int, d_model_guess: int) -> float:
    input_bytes = seq_len * feature_count * 4
    hidden_bytes = seq_len * d_model_guess * 4 * 4
    return (input_bytes + hidden_bytes) * 1.3


def intelligent_batch_size(seq_len: int, feature_count: int, base_batch: int, d_model_guess: int = 256) -> int:
    if not torch.cuda.is_available():
        return base_batch
    props = torch.cuda.get_device_properties(0)
    total_mem = props.total_memory
    target_mem = total_mem * 0.85
    per_sample = estimate_per_sample_bytes(seq_len, feature_count, d_model_guess)
    max_batch_by_mem = max(1, int(target_mem // max(per_sample, 1)))
    new_batch = max(base_batch, max_batch_by_mem)
    new_batch = min(new_batch, EFFECTIVE_BATCH_SIZE)
    return max(2, new_batch)


def compute_class_weights(targets: np.ndarray, cols: list[int]) -> torch.Tensor:
    rates = targets[:, cols].mean(axis=0)
    weights = ((1 - rates) / (rates + 1e-6)).astype(np.float32)
    return torch.from_numpy(weights)


def set_seed(seed: int | None) -> None:
    """Best-effort reproducibility aligned with 07_tri_task_nexus notebook."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
