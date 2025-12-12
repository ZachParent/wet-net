# Migration Inconsistencies Report: mai-bda → wet-net

This report details differences between the original `07_tri_task_nexus.py` notebook and the `wet-net` refactored codebase that may be causing worse model performance.

---

## 🔴 Critical Issues (Likely Performance Impact)

### 1. **Stratified Subsampling Logic is Different** 
**Location:** `TimeSeriesDataset._stratified_subsample()`

| mai-bda (original) | wet-net (new) |
|---|---|
| Keeps **all positive samples** if fewer than `max_samples`, fills remaining with negatives | Preserves **original class ratio** with complex clamping logic |

**Original logic (mai-bda):**
```python
if len(pos_idx) >= max_samples:
    chosen_pos = rng.choice(pos_idx, max_samples, replace=False)
    chosen = chosen_pos
else:
    remaining = max_samples - len(pos_idx)
    chosen_neg = rng.choice(neg_idx, min(len(neg_idx), remaining), replace=False)
    chosen = np.concatenate([pos_idx, chosen_neg])
```

**New logic (wet-net):**
```python
pos_frac = len(pos_idx) / total
pos_take = int(round(max_samples * pos_frac))
pos_take = max(1, min(len(pos_idx), pos_take))
neg_take = max_samples - pos_take
# ... complex overflow/deficit handling
```

**Impact:** The original prioritizes keeping **all anomalous samples**, which is crucial for imbalanced anomaly detection. The new approach may **undersample anomalies** when they are already rare.

**Recommendation:** ⚠️ **Restore original logic** — anomaly detection benefits from keeping all positive samples.

---

### 2. **Training Loop Loss Aggregation and Metrics Recording**
**Location:** `run_epoch()` in [loops.py](src/wet_net/training/loops.py)

| mai-bda (original) | wet-net (new) |
|---|---|
| Per-task loss tracked with simple `sums[key] += value` | Loss recording is **broken**: only the **last** task's loss gets recorded due to loop variable shadowing |

**The Bug:**
```python
for key, value in result["losses"].items():
    w = weights.get(key, 1.0)
    total = total + w * value
    total_weight += w
    weighted_losses.append(w * value)
if total_weight > 0:
    total = total / total_weight
    weighted_losses = [wl / total_weight for wl in weighted_losses]
    sums[key] += float(value.item())  # ❌ BUG: 'key' and 'value' are loop variables from the for-loop!
```

The line `sums[key] += float(value.item())` is **outside** the for-loop but uses `key` and `value`, so only the **last task** from the dictionary iteration is recorded. This means loss tracking per task is incorrect.

**Recommendation:** 🔧 **Fix immediately** — move the `sums[key]` update inside the loop:
```python
for key, value in result["losses"].items():
    w = weights.get(key, 1.0)
    total = total + w * value
    total_weight += w
    weighted_losses.append(w * value)
    sums[key] += float(value.item())  # ✅ Inside loop
```

---

### 3. **Loss Normalization by Weight Sum**
**Location:** `run_epoch()` in [loops.py](src/wet_net/training/loops.py#L113-L119)

| mai-bda (original) | wet-net (new) |
|---|---|
| Total loss = sum of `weight * loss` | Total loss = sum of `weight * loss` **divided by total_weight** |

**Original:**
```python
total = 0.0
for key, value in result["losses"].items():
    total += TASK_WEIGHTS.get(key, 1.0) * value
```

**New:**
```python
total = total / total_weight  # Extra normalization
weighted_losses = [wl / total_weight for wl in weighted_losses]
```

**Impact:** This changes the loss scale, which affects:
- Learning rate effective scale
- Gradient magnitudes
- Early stopping thresholds (`min_delta_abs=1e-4`)

**Recommendation:** 🔧 **Remove the `/ total_weight` normalization** to match original behavior, OR adjust learning rates and thresholds accordingly.

---

### 4. **`ensure_anomaly_coverage()` Signature Changed**
**Location:** [datasets.py](src/wet_net/data/datasets.py#L192-L221)

| mai-bda (original) | wet-net (new) |
|---|---|
| `ensure_anomaly_coverage(meta_df, splits)` | `ensure_anomaly_coverage(meta_df, splits, min_ratio=0.0, max_transfer=None)` |

The wet-net version accepts extra parameters that can limit anomaly transfer. If these are not set correctly when calling, you may end up with fewer anomalies in val/test sets.

**More importantly**, the transfer logic differs:
- Original: Moves **one** sample per target set if needed
- New: Can move **multiple** samples based on `min_ratio`

**Recommendation:** Verify calls use compatible defaults. Current defaults (`min_ratio=0.0`) should be safe.

---

## 🟠 Medium Issues (May Affect Performance)

### 5. **Default `MAX_SAMPLES` Calculation Differs**
**Location:** `max_samples_for_seq()` in [utils.py](src/wet_net/training/utils.py#L23-L26)

| mai-bda (original) | wet-net (new) |
|---|---|
| `MAX_SAMPLES = 80_000` (fixed global) | `base = 120_000` (higher) |

**Original:**
```python
MAX_SAMPLES = 80_000
```

**New:**
```python
def max_samples_for_seq(seq_len: int) -> int:
    base = 120_000
    scaled = int(base * 96 / max(96, seq_len))
    return max(8_000, min(150_000, scaled))
```

For `seq_len=96`: returns `120,000` (vs `80,000` original)

**Impact:** Potentially training on **50% more data**, which changes training dynamics. However, combined with Issue #1, you might paradoxically have **fewer anomalies** in proportion.

**Recommendation:** Consider using the original `80,000` value for reproducibility tests.

---

### 6. **`intelligent_batch_size()` Uses Different `d_model_guess` Default**
**Location:** [utils.py](src/wet_net/training/utils.py#L34)

| mai-bda (original) | wet-net (new) |
|---|---|
| `MAX_GRID_D_MODEL = max(cfg["d_model"] for cfg in GRID_SEARCH_CONFIGS)` = **612** | `d_model_guess=256` (hardcoded default) |

**Impact:** Memory estimation uses a much smaller model assumption, potentially allowing larger batches than the original would. This could affect gradient noise characteristics.

**Recommendation:** Pass the actual `d_model` or use a more conservative default.

---

### 7. **HORIZONS List Difference**
**Location:** [tri_task.py](src/wet_net/config/tri_task.py#L14)

| mai-bda (original) | wet-net (new) |
|---|---|
| `HORIZONS = [24, 48, 168, 336]` | Same ✅ |

This is **correct**, but worth double-checking since it's critical for target computation.

---

### 8. **SEQ_LENGTHS Default Values**
**Location:** [tri_task.py](src/wet_net/config/tri_task.py#L13)

| mai-bda (original) | wet-net (new) |
|---|---|
| `SEQ_LENGTHS = [96, 192, 360, 720, 1440, 2880, 4320]` | `SEQ_LENGTHS = [48, 96, 192, 360, 720, 1440]` |

**Difference:** 
- wet-net adds `48` 
- wet-net removes `2880` and `4320`

**Impact:** If you're testing with sequence lengths not in the new list, configs won't be found.

---

### 9. **Best Config Values Differ**
**Location:** [tri_task.py](src/wet_net/config/tri_task.py#L32-L49)

The `BEST_CONFIGS` in wet-net are **hardcoded** and may not match the grid search results from the original notebook. For example:

For `seq_len=96, optimize_for="recall"`:
- **wet-net**: `d_model=160, n_layers=4, n_heads=8, dropout=0.05, schedule="extended", pcgrad=True`
- **Original grid**: Dynamically determined from actual grid search

**Recommendation:** Re-run grid search or verify these match your original best results.

---

## 🟡 Minor Issues (Unlikely but Worth Noting)

### 10. **VIB `beta` Default Changed**
**Location:** [tri_task.py](src/wet_net/pipelines/tri_task.py#L209)

| mai-bda (original) | wet-net (new) |
|---|---|
| `"beta": 1e-3` | `"beta": 7.5e-4` |

**Impact:** Lower beta = weaker KL regularization, potentially less disentanglement.

---

### 11. **Missing `rolling_mean_24h` dtype cast**
**Location:** [preprocess.py](src/wet_net/data/preprocess.py#L115)

| mai-bda (original) | wet-net (new) |
|---|---|
| No explicit `.astype(np.float32)` on `rolling_mean_24h` | Same ❌ |

Both versions are missing this cast, but it's inconsistent with other features. Should be fine since the group transform should preserve dtype, but worth verifying.

---

### 12. **`set_seed()` Differences**
**Location:** [utils.py](src/wet_net/training/utils.py#L52-L61)

| mai-bda (original) | wet-net (new) |
|---|---|
| Sets `torch.backends.cudnn.deterministic = False` and `benchmark = True` | Does not set these |

**Impact:** Different CUDNN behavior could lead to slight non-determinism.

---

### 13. **Early Stopping Threshold Logic**
**Location:** [loops.py](src/wet_net/training/loops.py#L260-L275)

The wet-net version has more sophisticated early stopping with `min_delta_rel` (relative delta) in addition to absolute delta. The default is `min_delta_rel=0.0`, so it should match, but the combination with Issue #3 (normalized loss) means the effective threshold may differ.

---

## 📋 Summary of Recommended Actions

| Priority | Issue | Action |
|----------|-------|--------|
| 🔴 P0 | Loss tracking bug (#2) | Fix the loop variable scope immediately |
| 🔴 P0 | Loss normalization (#3) | Remove `/ total_weight` or adjust LR |
| 🔴 P1 | Stratified sampling (#1) | Restore original prioritize-positives logic |
| 🟠 P2 | Max samples (#5) | Test with `80,000` to match original |
| 🟠 P2 | Best configs (#9) | Verify or re-run grid search |
| 🟡 P3 | VIB beta (#10) | Use `1e-3` for reproduction |

---

## 🧪 Suggested Verification Tests

1. **Print anomaly ratios** after subsampling in both codebases — they should match
2. **Log individual task losses** per epoch to verify recording is correct
3. **Compare total loss values** at epoch 1 with same seed — should be identical
4. **Check gradient norms** after first backward pass — the normalization may cause differences

---

*Report generated by analyzing `07_tri_task_nexus.py` vs `wet-net/src/wet_net/*`*
