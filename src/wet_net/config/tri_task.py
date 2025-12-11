"""
Central config for the WetNet tri-task pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

SEQ_LENGTHS = [48, 96, 192, 360, 720, 1440]
HORIZONS = [24, 48, 168, 336]
FORECAST_HORIZON = 24
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
PCGRAD_GLOBAL_OPTIONS = [True, False]
METRIC_THRESHOLDS = [0.4, 0.5, 0.6, 0.7]


@dataclass
class ModelConfig:
    d_model: int
    n_layers: int
    n_heads: int
    dropout: float
    schedule_variant: str
    pcgrad: bool
    threshold: float


# Derived from the original grid + augmented metrics (best across thresholds)
BEST_CONFIGS: dict[str, dict[int, ModelConfig]] = {
    "recall": {
        48: ModelConfig(64, 2, 4, 0.15, "baseline", False, 0.4),
        96: ModelConfig(160, 4, 8, 0.05, "extended", True, 0.4),
        192: ModelConfig(160, 4, 8, 0.05, "extended", False, 0.4),
        360: ModelConfig(160, 4, 8, 0.05, "extended", True, 0.4),
        720: ModelConfig(160, 4, 8, 0.05, "extended", False, 0.4),
        1440: ModelConfig(96, 3, 4, 0.2, "baseline", False, 0.4),
    },
    "false_alarm": {
        48: ModelConfig(64, 2, 4, 0.15, "baseline", False, 0.7),
        96: ModelConfig(160, 4, 8, 0.05, "extended", False, 0.7),
        192: ModelConfig(160, 4, 8, 0.05, "extended", True, 0.7),
        360: ModelConfig(160, 4, 8, 0.05, "extended", True, 0.7),
        720: ModelConfig(96, 3, 4, 0.2, "extended", True, 0.7),
        1440: ModelConfig(96, 3, 4, 0.2, "baseline", False, 0.7),
    },
}


def get_best_config(seq_len: int, optimize_for: str) -> ModelConfig:
    if optimize_for not in BEST_CONFIGS:
        raise ValueError(f"optimize_for must be one of {list(BEST_CONFIGS.keys())}")
    if seq_len not in BEST_CONFIGS[optimize_for]:
        available = list(BEST_CONFIGS[optimize_for].keys())
        raise ValueError(f"No cached config for seq_len={seq_len}. Available: {available}")
    base = BEST_CONFIGS[optimize_for][seq_len]
    return apply_yaml_override(base, optimize_for, seq_len)


DEFAULT_YAML_PATH = Path(__file__).resolve().parent / "best_configs.yaml"


def apply_yaml_override(base: ModelConfig, optimize_for: str, seq_len: int, path: Path | None = None) -> ModelConfig:
    path = path or DEFAULT_YAML_PATH
    if not path.exists():
        return base
    try:
        overrides: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError:
        return base
    opt_section = overrides.get(optimize_for, {})
    seq_section = opt_section.get(str(seq_len)) or opt_section.get(int(seq_len))  # allow int keys
    if not isinstance(seq_section, dict):
        return base
    updated = base
    for field, value in seq_section.items():
        if hasattr(updated, field):
            updated = replace(updated, **{field: value})
    return updated
