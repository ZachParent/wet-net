from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wet_net.models.vib import VIBTransformer
from wet_net.models.wetnet import WetNet


def probability_from_zscore(z, midpoint=0.5, scale=1.0):
    return torch.sigmoid((z - midpoint) / scale)


def mass_from_reconstruction(zscore):
    prob_tensor = probability_from_zscore(zscore)
    prob = float(prob_tensor.item())
    theta = 0.05
    norm_mass = (1 - prob) * (1 - theta)
    return {"A": prob * (1 - theta), "N": norm_mass, "Theta": theta}


def mass_from_forecast(prob):
    prob = float(max(0.0, min(1.0, prob)))
    return {"A": prob, "N": 1 - prob, "Theta": 0.0}


def mass_from_uncertainty(uncert):
    uncert = float(max(0.0, min(1.0, uncert)))
    return {"A": 0.0, "N": 1 - uncert, "Theta": uncert}


def combine_masses(m1, m2):
    def intersect(h1, h2):
        if h1 == "Theta":
            return h2
        if h2 == "Theta":
            return h1
        if h1 == h2:
            return h1
        return None

    conflict = 0.0
    combined = {"A": 0.0, "N": 0.0, "Theta": 0.0}
    for h1, v1 in m1.items():
        for h2, v2 in m2.items():
            inter = intersect(h1, h2)
            if inter is None:
                conflict += v1 * v2
            else:
                combined[inter] += v1 * v2
    if conflict >= 1.0:
        return combined, conflict
    norm = 1 - conflict
    for key in combined:
        combined[key] /= norm
    return combined, conflict


def entropy_from_logvar(logvars):
    return torch.sigmoid((logvars - logvars.mean()) / (logvars.std() + 1e-6))


def fuse_probabilities(
    model: WetNet,
    vib_model: VIBTransformer,
    loader: DataLoader,
    recon_mean: float,
    recon_std: float,
    device: torch.device,
):
    conflict_scores = []
    fused_prob = []
    with torch.no_grad():
        for seq, _, _ in loader:
            seq = seq.to(device)
            outputs = model(seq)
            logits = torch.cat([outputs["short_logits"], outputs["long_logits"]], dim=-1)
            recon = F.mse_loss(outputs["reconstruction"], seq, reduction="none").mean(dim=(1, 2))
            z = (recon.cpu() - recon_mean) / recon_std
            recon_mass = [mass_from_reconstruction(val) for val in z]
            fore_probs = torch.sigmoid(logits)[:, 0].cpu().numpy()
            fore_mass = [mass_from_forecast(p) for p in fore_probs]
            vib_recon, vib_logits, c_mu, c_logvar, s_mu, s_logvar = vib_model(seq)
            avg_logvar = torch.mean(torch.cat([c_logvar, s_logvar], dim=-1), dim=(1, 2))
            uncert = entropy_from_logvar(avg_logvar).cpu().numpy()
            vib_mass = [mass_from_uncertainty(u) for u in uncert]
            for rm, fm, vm in zip(recon_mass, fore_mass, vib_mass, strict=False):
                comb, conflict = combine_masses(rm, fm)
                comb, conflict = combine_masses(comb, vm)
                fused_prob.append(comb["A"])
                conflict_scores.append(conflict)
    return np.array(fused_prob), np.array(conflict_scores)
