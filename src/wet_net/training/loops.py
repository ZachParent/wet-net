from __future__ import annotations

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from wet_net.models.wetnet import WetNet
from wet_net.training.utils import EFFECTIVE_BATCH_SIZE

TASK_WEIGHTS = {"reconstruction": 1.0, "forecast": 0.6, "short": 1.2, "long": 1.2}
console = Console()


def pcgrad_step(model: torch.nn.Module, objectives: list[torch.Tensor], scale: float = 1.0) -> None:
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
            dot = sum((g_i * g_j).sum() for g_i, g_j in zip(projected[i], projected[j], strict=False))
            if dot < 0:
                norm_sq = sum((g_j**2).sum() for g_j in projected[j]) + 1e-12
                coeff = dot / norm_sq
                projected[i] = [g_i - coeff * g_j for g_i, g_j in zip(projected[i], projected[j], strict=False)]
    for idx, (p, grad_components) in enumerate(zip(params, zip(*projected, strict=False), strict=False)):
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
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    active_tasks: list[str],
    pos_weight_short: torch.Tensor,
    pos_weight_long: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
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
    active_tasks: list[str],
    pos_weight_short: torch.Tensor,
    pos_weight_long: torch.Tensor,
    device: torch.device,
    train: bool,
    use_pcgrad: bool,
    effective_batch_size: int = EFFECTIVE_BATCH_SIZE,
    task_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()
    aggregates = {"total": 0.0, "count": 0}
    recall_counts = {
        "short_tp": 0,
        "short_pos": 0,
        "long_tp": 0,
        "long_pos": 0,
    }
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
            with autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                enabled=torch.cuda.is_available(),
            ):
                result = forward_pass(model, batch, active_tasks, pos_weight_short, pos_weight_long, device)
                total = 0.0
                total_weight = 0.0
                weighted_losses = []
                weights = task_weights or TASK_WEIGHTS
                for key, value in result["losses"].items():
                    w = weights.get(key, 1.0)
                    total = total + w * value
                    total_weight += w
                    # keep per-task contribution scaled for pcgrad if enabled
                    weighted_losses.append(w * value)
                # normalize by sum of active weights to keep loss scale comparable
                if total_weight > 0:
                    total = total / total_weight
                    weighted_losses = [wl / total_weight for wl in weighted_losses]
                    sums[key] += float(value.item())
                # simple recall metrics for classification heads
                targets = result["targets"]
                if "short" in active_tasks and "short_logits" in result["outputs"]:
                    logits = result["outputs"]["short_logits"]
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    t = targets[:, : pos_weight_short.shape[0]]
                    recall_counts["short_tp"] += int(((preds == 1) & (t == 1)).sum().item())
                    recall_counts["short_pos"] += int((t == 1).sum().item())
                if "long" in active_tasks and "long_logits" in result["outputs"]:
                    logits = result["outputs"]["long_logits"]
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    t = targets[:, -pos_weight_long.shape[0] :]
                    recall_counts["long_tp"] += int(((preds == 1) & (t == 1)).sum().item())
                    recall_counts["long_pos"] += int((t == 1).sum().item())
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
    # recall summaries
    if recall_counts["short_pos"] > 0:
        metrics["recall_short"] = recall_counts["short_tp"] / recall_counts["short_pos"]
    if recall_counts["long_pos"] > 0:
        metrics["recall_long"] = recall_counts["long_tp"] / recall_counts["long_pos"]
    return metrics


def build_training_stages(schedule_variant: str, pcgrad_enabled: bool) -> list[dict]:
    variants = {
        "baseline": (10, 16, 24),
        "extended": (14, 22, 32),
    }
    e1, e2, e3 = variants[schedule_variant]
    return [
        {
            "name": f"{schedule_variant}_stage1",
            "epochs": e1,
            "lr": 3e-4,
            "tasks": ["reconstruction"],
            "pcgrad": False,
            "patience": 3,
        },
        {
            "name": f"{schedule_variant}_stage2",
            "epochs": e2,
            "lr": 2.5e-4,
            "tasks": ["reconstruction", "forecast"],
            "pcgrad": pcgrad_enabled,
            "patience": 4,
        },
        {
            "name": f"{schedule_variant}_stage3",
            "epochs": e3,
            "lr": 2e-4,
            "tasks": ["reconstruction", "forecast", "short", "long"],
            "pcgrad": pcgrad_enabled,
            "patience": 5,
        },
    ]


def train_staged_model(
    model: WetNet,
    loaders: dict[str, DataLoader],
    stages: list[dict],
    pos_weight_short: torch.Tensor,
    pos_weight_long: torch.Tensor,
    device: torch.device,
    early_stop: bool = True,
    min_delta_abs: float = 1e-4,
    min_delta_rel: float = 0.0,
    task_weights: dict[str, float] | None = None,
    monitor: str = "total",
) -> list[dict]:
    history: list[dict] = []
    for stage in stages:
        optimizer = AdamW(model.parameters(), lr=stage["lr"])
        best_val = float("inf")
        best_state = None
        wait = 0
        patience = stage.get("patience", stage["epochs"]) if early_stop else None
        tasks_label = "/".join(stage["tasks"])
        console.rule(f"{stage['name']} | tasks={tasks_label} | lr={stage['lr']} | pcgrad={stage['pcgrad']}")
        with Progress(
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} ep"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as stage_progress:
            stage_task = stage_progress.add_task(f"{stage['name']}", total=stage["epochs"])
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
                    task_weights=task_weights,
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
                    task_weights=task_weights,
                )
                history.append({"stage": stage["name"], "epoch": epoch, "split": "train", **train_metrics})
                history.append({"stage": stage["name"], "epoch": epoch, "split": "val", **val_metrics})
                weights = task_weights or TASK_WEIGHTS

                def comp_str(metrics: dict[str, float], current_weights: dict[str, float]) -> str:
                    parts = []
                    for key in ("reconstruction", "forecast", "short", "long"):
                        lk = f"loss_{key}"
                        if lk in metrics:
                            parts.append(f"{key[:5]}={metrics[lk]:.4f}(w={current_weights.get(key, 1.0):.2f})")
                        else:
                            parts.append(f"{key[:5]}=-- (w={current_weights.get(key, 1.0):.2f})")
                    if "recall_short" in metrics:
                        parts.append(f"rS={metrics['recall_short']:.3f}")
                    if "recall_long" in metrics:
                        parts.append(f"rL={metrics['recall_long']:.3f}")
                    return " ".join(parts)

                msg = (
                    f"[stage {stage['name']}] ep {epoch + 1}/{stage['epochs']} "
                    f"train_tot={train_metrics['loss_total']:.4f} [{comp_str(train_metrics, weights)}] "
                    f"val_tot={val_metrics['loss_total']:.4f} [{comp_str(val_metrics, weights)}]"
                )
                console.log(msg)
                # choose metric to monitor
                val_metric = val_metrics["loss_total"]
                if monitor == "cls" and ("loss_short" in val_metrics or "loss_long" in val_metrics):
                    cls_terms = []
                    if "loss_short" in val_metrics:
                        cls_terms.append(val_metrics["loss_short"])
                    if "loss_long" in val_metrics:
                        cls_terms.append(val_metrics["loss_long"])
                    if cls_terms:
                        val_metric = sum(cls_terms) / len(cls_terms)
                delta_needed = max(min_delta_abs, abs(best_val) * min_delta_rel)
                if val_metric < best_val - delta_needed:
                    best_val = val_metric
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if early_stop and patience and wait >= patience:
                        console.log(
                            f"[stage {stage['name']}] early stopping after {epoch + 1} epochs "
                            f"(no val improvement for {patience} epochs; best={best_val:.4f})."
                        )
                        break
                stage_progress.advance(stage_task, 1)
        if best_state:
            model.load_state_dict(best_state)
    return history
