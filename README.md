# WetNet

WetNet helps Aigües de Barcelona (Barcelona Water Company) predict anomalous water consumption, using machine learning.

![PyPI - Version](https://img.shields.io/pypi/v/wet-net)
![PyPI - License](https://img.shields.io/pypi/l/wet-net)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/wet-net)

![WetNet logo](https://github.com/ZachParent/wet-net/raw/main/assets/wet-net-logo.jpeg)

## Setup

### 1. Install uv

Our project uses uv to manage dependencies in a reproducible way. See [Installing uv](https://docs.astral.sh/uv/getting-started/installation/) documentation for installation instructions.

> [!TIP]
> You can skip the rest of the setup if you just want to run the scripts and see the project in action! Run `uvx wet-net --help` to see the available commands in the CLI. This will install the latest release of the project from PyPI in an isolated virtual environment.

### 2. Clone the repository

```bash
git clone https://github.com/ZachParent/wet-net.git
cd wet-net
```

### 3. Install the project dependencies

```bash
uv sync --locked
```

> Torch install hint (if not pulled via lockfile):
> - CPU-only: `pip install torch torchvision torchaudio`
> - CUDA 12.8 (recommended for GPUs): `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`

### HuggingFace token (only needed if you plan to push trained models)
- Copy `.env.template` to `.env` and set `HF_TOKEN=<your_token>`.
- Load and export: `source .env && uv run wet-net <command>`.
- Alternative: `huggingface-cli login` (persists token).

To verify your HuggingFace token and check repository access permissions:
```bash
# Load .env and check token validity
source .env && uv run wet-net hf-check WetNet/wet-net

# Use a custom environment variable
uv run wet-net hf-check WetNet/wet-net --env-var MY_HF_TOKEN
```

End-to-end train (full data) + push example:
```bash
# assuming data is preprocessed and CUDA available
source .env && uv run wet-net train --seq-len 96 --optimize-for recall
uv run python - <<'PY'
import os, torch
from huggingface_hub import HfApi, HfFolder
from wet_net.paths import RESULTS_DIR
run_dir = RESULTS_DIR / "wetnet" / "seq96_recall"
repo = os.environ.get("HF_REPO", "your-username/wetnet-seq96-recall")
api = HfApi()
api.create_repo(repo, exist_ok=True)
api.upload_file(path_or_fileobj=run_dir / "wetnet.pt", path_in_repo="wetnet.pt", repo_id=repo)
api.upload_file(path_or_fileobj=run_dir / "vib.pt", path_in_repo="vib.pt", repo_id=repo)
api.upload_file(path_or_fileobj=run_dir / "config.json", path_in_repo="config.json", repo_id=repo)
PY
```
Make sure `HUGGINGFACE_TOKEN` is set in your environment before running the upload step.

## Quick Start

### 1. Try it out immediately (no setup required)

```bash
# Install and run the CLI directly from PyPI
uvx wet-net --help

# Run a complete training pipeline with mock data
uvx wet-net pre-process --mock && uvx wet-net train --seq-len 48 --fast --mock
```

### 2. Full local setup for development

```bash
# Clone and set up the development environment
git clone https://github.com/ZachParent/wet-net.git
cd wet-net
uv sync --locked
```

## Usage

### Run the scripts

The CLI entry points target the WetNet tri-task pipeline with cached best configs per sequence length:

```bash
# Generate data (real: pass your private URL; mock: bundled synthetic)
uv run wet-net pre-process --data-url https://<your-parquet-or-zip-url>   # real data
uv run wet-net pre-process --mock                                         # synthetic data

# Train with production settings
uv run wet-net train --seq-len 192 --optimize-for recall
uv run wet-net train --seq-len 192 --optimize-for false_alarm

# Train with fast settings for testing (use --fast and --mock)
uv run wet-net train --seq-len 48 --fast --mock --run-suffix _test

# Optional: push trained artifacts to Hugging Face (requires .env sourced)
source .env && uv run wet-net train --seq-len 96 --optimize-for recall --push-to-hub --hub-model-name WetNet/wet-net

# Optional: override hyperparameters by editing src/wet_net/config/best_configs.yaml
# (any fields you change there are picked up automatically at runtime)

# Evaluate pre-trained models (automatically loaded from HuggingFace)
uv run wet-net evaluate --seq-len 192 --optimize-for recall

# Evaluate with custom local checkpoints
uv run wet-net evaluate --seq-len 192 --optimize-for recall --local
```

### Development and Testing

For rapid development and testing, use the fast settings that mirror the mini notebook experiment:

```bash
# Complete development workflow with tiny experiment
uv run wet-net pre-process --mock
uv run wet-net train --seq-len 48 --fast --mock --max-epochs 2 --run-suffix _tiny
uv run wet-net evaluate --seq-len 48 --run-suffix _tiny --mock
```

This matches the small notebook experiment and writes results to a separate directory to avoid overwriting production artifacts.

### Full Training Pipeline (Production)

⚠️ **Warning**: The following command trains models for ALL sequence lengths and BOTH optimization objectives. This is an extremely long-running process (potentially several days depending on hardware) and should only be run if you need comprehensive model coverage.

```bash
# Train complete set of models for all sequence lengths and objectives
# This will take a very long time - ensure you have stable power and sufficient GPU resources
for opt in recall false_alarm; do
  for s in 48 96 192 360 720 1440; do
    uv run wet-net train \
      --seq-len $s \
      --optimize-for $opt \
      --min-anomaly-ratio 0.05 \
      --early-stop-metric cls \
      --recon-weight 0.3 \
      --forecast-weight 0.5 \
      --short-weight 2 \
      --long-weight 2
  done
done
```

**Recommended approach**: Start with a single sequence length to test the pipeline:
```bash
# Test with just one sequence length first
uv run wet-net train --seq-len 192 --optimize-for recall \
  --min-anomaly-ratio 0.05 --early-stop-metric cls \
  --recon-weight 0.3 --forecast-weight 0.5 --short-weight 2 --long-weight 2
```

### Model Repository

All pre-trained WetNet models are hosted in the official HuggingFace repository:

**🤗 [https://huggingface.co/WetNet/wet-net](https://huggingface.co/WetNet/wet-net)**

The repository contains trained models for all sequence lengths (48, 96, 192, 360, 720, 1440) optimized for both objectives (recall, false_alarm). By default, the `wet-net evaluate` command automatically downloads and loads the appropriate model from this repository.

#### Available Models
- **Sequence Lengths**: 48, 96, 192, 360, 720, 1440
- **Optimization Objectives**: `recall` (maximize anomaly detection), `false_alarm` (minimize false positives)
- **Model Files**: `wetnet.pt` (main model), `vib.pt` (uncertainty probe), `config.json` (hyperparameters)

#### Evaluation Modes
```bash
# Default: Automatically load pre-trained model from HuggingFace
uv run wet-net evaluate --seq-len 192 --optimize-for recall

# Force use of local checkpoints (if available)
uv run wet-net evaluate --seq-len 192 --optimize-for recall --local
```

### HuggingFace Token Management

The `hf-check` command helps you verify your HuggingFace authentication and repository access:

```bash
# Check token validity and repository access
uv run wet-net hf-check WetNet/wet-net

# Use a specific environment variable
uv run wet-net hf-check WetNet/wet-net --env-var CUSTOM_HF_TOKEN

# The command displays:
# - Token authentication status
# - User and organization information
# - Repository visibility (public/private)
# - Inferred read/write permissions
```

The code for the scripts can be found in the [src/wet_net/scripts](src/wet_net/scripts) directory.

### Tri-Task WetNet

- Model: Cyclic-UniTS backbone + 4 heads (reconstruction, 24h forecast, short/long anomaly logits), class `wet_net.models.wetnet.WetNet`.
- Cached best configs: one per sequence length (96, 192, 360, 720, 1440) for two objectives (recall, false_alarm). Each stores schedule variant, PCGrad flag, transformer width/depth/heads/dropout, and the fusion threshold to report.
- VIB fusion: lightweight dual-VIB probe is trained after the main model to provide uncertainty mass for Dempster–Shafer fusion.
- Data: preprocessing lives in `wet_net.data.preprocess`. For real runs supply the parquet URL via `--data-url` or `WETNET_DATA_URL`. A mock parquet is shipped so the full pipeline can run without private data.

### Notebooks

We use Jupyter notebooks to show the process of preparing the data, training the model, and evaluating the model, with descriptions and code. You can run the notebooks by opening them up in your favorite IDE. Be sure to choose the `.venv` kernel which is created and managed by uv. The notebooks can be found in the [notebooks](notebooks) directory.

### 01_pre_process.ipynb

The [01_pre_process.ipynb](notebooks/01_pre_process.ipynb) notebook shows the process of preparing the data. It includes:

- Loading the data
- Pre-processing the data
- Saving the pre-processed data

### 02_train.ipynb

The [02_train.ipynb](notebooks/02_train.ipynb) notebook shows the process of training the model. It includes:

- Loading the pre-processed data
- Training the model
- Saving the trained model

### 03_evaluate.ipynb

The [03_evaluate.ipynb](notebooks/03_evaluate.ipynb) notebook shows the process of evaluating the model. It includes:

- Loading the trained model
- Evaluating the model
- Saving the evaluation results

## Contributing

We welcome contributions to the project. Please feel free to submit an issue or pull request.

For more information on contributing, including how to set up pre-commit hooks and how to cut a new release, see [CONTRIBUTING.md](CONTRIBUTING.md).
