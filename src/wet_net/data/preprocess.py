"""
Data download + preprocessing utilities for the WetNet tri‑task pipeline.

Key behaviors:
- If --mock is passed (or mock=True), a tiny synthetic parquet is generated so
  the whole pipeline can run without proprietary data.
- For real runs we expect the caller to provide the download URL (cannot be
  published here). We fail fast with a clear message when it is missing.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Default feature columns (mirrors the notebook)
FEATURE_COLS: list[str] = [
    "CONSUMO_REAL_norm",
    "rolling_mean_24h",
    "rolling_std_24h",
    "lag_1",
    "lag_24",
    "diff_1",
    "diff_24",
    "is_missing",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "is_weekend",
]

# Default paths relative to current working directory
RAW_PARQUET = Path("./data/raw/anomalous_consumption.parquet")
PROCESSED_PARQUET = Path("./data/processed/anomalous_consumption_preprocessed.parquet")
MOCK_PARQUET = Path("./data/mock/mock_consumption.parquet")
MOCK_CSV = Path("./data/mock/mock_consumption.csv")


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    return dest


def create_mock_dataset(out_path: Path = MOCK_PARQUET, days: int = 30, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2022-01-01", periods=24 * days, freq="H")
    policies = ["A100", "B200"]
    rows = []
    for pol in policies:
        base = rng.normal(loc=120, scale=20, size=len(timestamps))
        # inject anomalies
        anomaly_hours = rng.choice(len(timestamps), size=max(4, days // 2), replace=False)
        consum = base.copy()
        consum[anomaly_hours] *= rng.uniform(1.8, 2.6, size=len(anomaly_hours))
        flags = np.isin(range(len(timestamps)), anomaly_hours)
        for t, c, is_anom in zip(timestamps, consum, flags, strict=False):
            rows.append(
                {
                    "POLISSA_SUBM": pol,
                    "FECHA_HORA": t,
                    "CONSUMO_REAL": c,
                    "is_anomalous": float(is_anom),
                }
            )
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def robust_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess time series data with feature engineering.
    This is a line-for-line twin of the notebook's implementation
    (mai_bda.data.preprocessing.robust_preprocessing).
    """
    data = df.copy()

    data["FECHA_HORA"] = pd.to_datetime(data["FECHA_HORA"])
    if "START_DATE" in data.columns:
        data["START_DATE"] = pd.to_datetime(data["START_DATE"])
        data["END_DATE"] = pd.to_datetime(data["END_DATE"])

    data = data.sort_values(["POLISSA_SUBM", "FECHA_HORA"]).reset_index(drop=True)

    data["hour_of_day"] = data["FECHA_HORA"].dt.hour
    data["day_of_week"] = data["FECHA_HORA"].dt.dayofweek
    data["hour_sin"] = np.sin(2 * np.pi * data["hour_of_day"] / 24).astype(np.float32)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour_of_day"] / 24).astype(np.float32)
    data["day_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7).astype(np.float32)
    data["day_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7).astype(np.float32)
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(np.float32)

    if "is_anomalous" not in data.columns and "START_DATE" in data.columns:
        data["is_anomalous"] = (
            (data["FECHA_HORA"] >= data["START_DATE"]) & (data["FECHA_HORA"] <= data["END_DATE"])
        ).astype(np.float32)
    elif "is_anomalous" not in data.columns:
        data["is_anomalous"] = 0.0

    if "CONSUMO_REAL" in data.columns:
        data["is_missing"] = data["CONSUMO_REAL"].isna().astype(np.float32)
        data["CONSUMO_REAL"] = (
            data.groupby("POLISSA_SUBM")["CONSUMO_REAL"]
            .transform(lambda g: g.interpolate(method="linear", limit=5, limit_direction="both"))
            .fillna(0.0)
        )
        mean = data["CONSUMO_REAL"].mean()
        std = data["CONSUMO_REAL"].std() + 1e-8
        data["CONSUMO_REAL_norm"] = ((data["CONSUMO_REAL"] - mean) / std).astype(np.float32)

        data["rolling_mean_24h"] = (
            data.groupby("POLISSA_SUBM")["CONSUMO_REAL_norm"].transform(lambda x: x.rolling(24, min_periods=1).mean())
        ).fillna(0.0)
        group = data.groupby("POLISSA_SUBM")["CONSUMO_REAL_norm"]
        data["lag_1"] = group.shift(1).fillna(0.0).astype(np.float32)
        data["lag_24"] = group.shift(24).fillna(0.0).astype(np.float32)
        data["diff_1"] = (data["CONSUMO_REAL_norm"] - data["lag_1"]).astype(np.float32)
        data["diff_24"] = (data["CONSUMO_REAL_norm"] - data["lag_24"]).astype(np.float32)
        rolling_std = group.transform(lambda x: x.rolling(24, min_periods=1).std()).fillna(0.0)
        data["rolling_std_24h"] = rolling_std.astype(np.float32)
    return data


def load_preprocessed_dataframe(
    path: Path | str = PROCESSED_PARQUET,
    *,
    force_reprocess: bool = False,
    use_cache: bool = True,
    data_url: str | None = None,
    mock: bool = False,
) -> pd.DataFrame:
    """
    Load (or build) the preprocessed parquet.

    - If `force_reprocess` is True, always re-run preprocessing from raw.
    - If `use_cache` is True and the processed file exists, reuse it.
    - Otherwise, rebuild from raw (downloading if needed).
    """
    path = Path(path)
    raw_path = MOCK_PARQUET if mock else RAW_PARQUET

    def _ensure_raw():
        if raw_path.exists():
            return
        if mock:
            create_mock_dataset(raw_path)
        else:
            if not data_url:
                raise FileNotFoundError(
                    f"Raw dataset not found at {raw_path} and no data_url provided. "
                    "Pass --data-url or set WETNET_DATA_URL."
                )
            output_path = raw_path if not str(data_url).lower().endswith(".zip") else raw_path.with_suffix(".zip")
            download_file(data_url, output_path)

    # Use cache if allowed and present
    if use_cache and path.exists() and not force_reprocess:
        return pd.read_parquet(path)

    _ensure_raw()
    df_raw = pd.read_parquet(raw_path)
    df_proc = robust_preprocessing(df_raw)
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        df_proc.to_parquet(path, index=False)
    return df_proc


def prepare_dataset(
    mock: bool = False,
    data_url: str | None = None,
    force_mock_regen: bool = False,
    force_reprocess: bool = False,
) -> Path:
    """
    Ensure a parquet is available and preprocessed.
    """
    raw_path = MOCK_PARQUET if mock else RAW_PARQUET
    processed_path = PROCESSED_PARQUET if not mock else Path("./data/processed/mock_preprocessed.parquet")

    if mock:
        if force_mock_regen or not raw_path.exists():
            create_mock_dataset(raw_path)
        elif MOCK_CSV.exists() and force_mock_regen:
            df_csv = pd.read_csv(MOCK_CSV, parse_dates=["FECHA_HORA"])
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            df_csv.to_parquet(raw_path, index=False)
    else:
        if not data_url:
            raise ValueError(
                "A data URL is required for real runs. Pass --data-url or set WETNET_DATA_URL (link cannot be shared)."
            )
        if not raw_path.exists():
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            if data_url.lower().endswith(".zip"):
                zip_path = raw_path.with_suffix(".zip")
                download_file(data_url, zip_path)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    parquet_members = [m for m in zf.namelist() if m.lower().endswith(".parquet")]
                    if not parquet_members:
                        raise ValueError("Zip archive does not contain any parquet files.")
                    member = parquet_members[0]
                    extract_dest = zip_path.parent
                    extract_dest.mkdir(parents=True, exist_ok=True)
                    zf.extract(member, extract_dest)
                    extracted_path = (extract_dest / member).resolve()
                    extracted_path.rename(raw_path)
            else:
                download_file(data_url, raw_path)

    if processed_path.exists() and not force_reprocess:
        return processed_path

    df_raw = pd.read_parquet(raw_path)
    df_proc = robust_preprocessing(df_raw)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df_proc.to_parquet(processed_path, index=False)
    return processed_path


def select_feature_columns(df: pd.DataFrame, allowed: Iterable[str] = FEATURE_COLS) -> list[str]:
    return [c for c in allowed if c in df.columns]
