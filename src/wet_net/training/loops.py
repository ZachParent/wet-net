from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from rich.progress import Progress

from wet_net.models.wetnet import WetNet
from wet_net.training.utils import EFFECTIVE_BATCH_SIZE

TASK_WEIGHTS = {"reconstruction": 1.0, "forecast": 0.6, "short": 1.2, "long": 1.2}


def pcgrad_step(model: torch.nn.Module, objectives: List[torch.Tensor], scale: float = 1.0) -> None:
    params = [p for p in model.parameters() if p.requires_grad]
    prev_grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
    grads = []
    for idx, obj in enumerate(objectives):
        model.zero_grad(set_to_none=True)
        obj.backward(retain_graph=(idx < len(objectives) - 1))
        grads.append([p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params])
    projected = [[g.clone() for g in grad_list] for grad_list in grads]
    for i in range(len(projected)):
        for j in range(len(projected)):
            if i == j:
                continue
            dot = sum((g_i * g_j).sum() for g_i, g_j in zip(projected[i], projected[j]))
            if dot < 0:
                norm_sq = sum((g_j**2).sum() for g_j in projected[j]) + 1e-12
                coeff = dot / norm_sq
                projected[i] = [g_i - coeff * g_j for g_i, g_j in zip(projected[i], projected[j])]
    for idx, (p, grad_components) in enumerate(zip(params, zip(*projected))):
        total_grad = torch.zeros_like(p)
        for g in grad_components:
            total_grad += g
        contribution = total_grad * scale
        prev = prev_grads[idx]
        if p.grad is None:
            p.grad = prev + contribution
        else:
            p.grad = prev + contribution


def forward_pass(
    model: WetNet,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    active_tasks: List[str],
    pos_weight_short: torch.Tensor,
    pos_weight_long: torch.Tensor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    seq, multi_targets, future = batch
    seq = seq.to(device)
    multi_targets = multi_targets.to(device)
    future = future.to(device)
    outputs = model(seq)
    losses = {}
    if "reconstruction" in active_tasks:
        losses["reconstruction"] = F.mse_loss(outputs["reconstruction"], seq)
    if "forecast" in active_tasks:
        losses["forecast"] = F.mse_loss(outputs["forecast"], future)
    if "short" in active_tasks:
        losses["short"] = F.binary_cross_entropy_with_logits(
            outputs["short_logits"],
            multi_targets[:, : pos_weight_short.shape[0]],
            pos_weight=pos_weight_short.to(device),
        )
    if "long" in active_tasks:
        losses["long"] = F.binary_cross_entropy_with_logits(
            outputs["long_logits"],
            multi_targets[:, -pos_weight_long.shape[0] :],
            pos_weight=pos_weight_long.to(device),
        )
    return {"losses": losses, "outputs": outputs, "seq": seq, "targets": multi_targets, "future": future}


def run_epoch(
    model: WetNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    active_tasks: List[str],
    pos_weight_short: torch.Tensor,
    pos_weight_long: torch.Tensor,
    device: torch.device,
    train: bool,
    use_pcgrad: bool,
    effective_batch_size: int = EFFECTIVE_BATCH_SIZE,
) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()
    aggregates = {"total": 0.0, "count": 0}
    loss_keys = ["reconstruction", "forecast", "short", "long"]
    sums = {k: 0.0 for k in loss_keys}
    accum_steps = 1
    scaler = None
    use_amp = False
    if train:
        accum_steps = max(1, effective_batch_size // loader.batch_size)
        optimizer.zero_grad(set_to_none=True)
        use_amp = torch.cuda.is_available() and (not use_pcgrad)
        scaler = GradScaler(device="cuda" if torch.cuda.is_available() else "cpu", enabled=use_amp)
    step_in_accum = 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=torch.cuda.is_available()):
                result = forward_pass(model, batch, active_tasks, pos_weight_short, pos_weight_long, device)
                total = 0.0
                weighted_losses = []
                for key, value in result["losses"].items():
                    w = TASK_WEIGHTS.get(key, 1.0)
                    total = total + w * value
                    weighted_losses.append(w * value)
                    sums[key] += float(value.item())
            if train:
                if use_pcgrad and len(weighted_losses) > 1:
                    pcgrad_step(model, weighted_losses, scale=1.0 / accum_steps)
                else:
                    scaled_loss = total / accum_steps
                    (scaler.scale(scaled_loss) if scaler else scaled_loss).backward()
                step_in_accum += 1
                if step_in_accum % accum_steps == 0:
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            aggregates["total"] += float(total.item())
            aggregates["count"] += 1
    if train and step_in_accum % accum_steps != 0 and step_in_accum > 0:
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    metrics = {"loss_total": aggregates["total"] / max(1, aggregates["count"])}
    for key in loss_keys:
        metrics[f"loss_{key}"] = sums[key] / max(1, aggregates["count"])
    return metrics


def build_training_stages(schedule_variant: str, pcgrad_enabled: bool) -> List[Dict]:
    variants = {
        "baseline": (10, 16, 24),
        "extended": (14, 22, 32),
    }
    e1, e2, e3 = variants[schedule_variant]
    return [
        {"name": f"{schedule_variant}_stage1", "epochs": e1, "lr": 3e-4, "tasks": ["reconstruction"], "pcgrad": False},
        {
            "name": f"{schedule_variant}_stage2",
            "epochs": e2,
            "lr": 2.5e-4,
            "tasks": ["reconstruction", "forecast"],
            "pcgrad": pcgrad_enabled,
        },
        {
            "name": f"{schedule_variant}_stage3",
            "epochs": e3,
            "lr": 2e-4,
            "tasks": ["reconstruction", "forecast", "short", "long"],
            "pcgrad": pcgrad_enabled,
        },
    ]


def train_staged_model(
    model: WetNet,
    loaders: Dict[str, DataLoader],
    stages: List[Dict],
    pos_weight_short: torch.Tensor,
    pos_weight_long: torch.Tensor,
    device: torch.device,
    progress: Optional[Progress] = None,
    progress_task: Optional[int] = None,
) -> List[Dict]:
    history: List[Dict] = []
    for stage in stages:
        optimizer = AdamW(model.parameters(), lr=stage["lr"])
        best_val = float("inf")
        best_state = None
        wait = 0
        patience = stage.get("patience", stage["epochs"])
        for epoch in range(stage["epochs"]):
            train_metrics = run_epoch(
                model,
                loaders["train"],
                optimizer,
                stage["tasks"],
                pos_weight_short,
                pos_weight_long,
                device,
                train=True,
                use_pcgrad=stage["pcgrad"],
            )
            val_metrics = run_epoch(
                model,
                loaders["val"],
                optimizer,
                stage["tasks"],
                pos_weight_short,
                pos_weight_long,
                device,
                train=False,
                use_pcgrad=False,
            )
            history.append({"stage": stage["name"], "epoch": epoch, "split": "train", **train_metrics})
            history.append({"stage": stage["name"], "epoch": epoch, "split": "val", **val_metrics})
            if val_metrics["loss_total"] < best_val - 1e-4:
                best_val = val_metrics["loss_total"]
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if patience and wait >= patience:
                    break
            if progress and progress_task is not None:
                progress.advance(progress_task, 1)
        if best_state:
            model.load_state_dict(best_state)
    return history
