# WetNet

WetNet helps Aigües de Barcelona (Barcelona Water Company) predict anomalous water consumption, using machine learning.

![WetNet logo](assets/wet-net-logo.jpeg)

## Setup

### 1. Install uv

Our project uses uv to manage dependencies in a reproducible way. You can install it by running the following command:
Visit [Installing uv](https://docs.astral.sh/uv/getting-started/installation/) documentation for installation instructions.

> [!TIP]
> You can skip the rest of the setup if you just want to run the scripts and see the project in action! Run `uvx https://github.com/ZachParent/wet-net.git --help` to see the available commands in the CLI.

### 2. Clone the repository

```bash
git clone https://github.com/ZachParent/wet-net.git
cd wet-net
```

### 3. Install the project dependencies

```bash
uv sync --locked
```

## Usage

### Run the scripts

The easiest way to verify that the project is working is to run the scripts. These command line interfaces include help documentation.

```bash
uv run wet-net pre-process --help
uv run wet-net train --help
uv run wet-net evaluate --help
```

The code for the scripts can be found in the [src/wet_net/scripts](src/wet_net/scripts) directory.

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

### Pre-commit hooks

We use pre-commit hooks to run checks on the code before it is committed. You can install the pre-commit hooks by running the following command in the root of the repository:

```bash
uv run pre-commit install
```
