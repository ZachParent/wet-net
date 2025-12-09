from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wet_net.models.wetnet import WetNet


def collect_predictions(model: WetNet, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    probs, targets, recon_errs = [], [], []
    forecasts, futures = [], []
    with torch.no_grad():
        for seq, multi_targets, future in loader:
            seq = seq.to(device)
            outputs = model(seq)
            logits = torch.cat([outputs["short_logits"], outputs["long_logits"]], dim=-1)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(multi_targets.numpy())
            err = F.mse_loss(outputs["reconstruction"], seq, reduction="none").mean(dim=(1, 2))
            recon_errs.append(err.cpu().numpy())
            forecasts.append(outputs["forecast"].cpu().numpy())
            futures.append(future.numpy())
    return {
        "probabilities": np.concatenate(probs, axis=0),
        "targets": np.concatenate(targets, axis=0),
        "recon_error": np.concatenate(recon_errs, axis=0),
        "forecast": np.concatenate(forecasts, axis=0),
        "future": np.concatenate(futures, axis=0),
    }


def build_prediction_frame(
    meta_df: pd.DataFrame, indices: list[int], predictions: dict[str, np.ndarray]
) -> pd.DataFrame:
    subset = meta_df.loc[indices].copy().reset_index(drop=True)
    return subset.assign(
        prob_24=predictions["probabilities"][:, 0],
        prob_48=predictions["probabilities"][:, 1],
        prob_168=predictions["probabilities"][:, 2],
        prob_336=predictions["probabilities"][:, 3],
        recon_error=predictions["recon_error"],
    )
