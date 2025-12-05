from pathlib import Path

import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader
from torchvision import datasets

from wet_net.mnist_demo.classifier import MNISTClassifierCNN
from wet_net.mnist_demo.transforms import MNIST_TRANSFORMS

app = typer.Typer()


@app.command()
def evaluate(
    data_dir: str = typer.Option("./data", help="Directory containing the dataset"),
    results_dir: str = typer.Option("./results", help="Directory to save evaluation results"),
    dry_run: bool = typer.Option(False, help="Dry run the evaluation."),
    hub_model_name: str = typer.Option("zachparent/mnist-classifier", help="Name of the model to evaluate"),
):
    """
    Evaluate the model from HuggingFace and save results to CSV.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        typer.secho(f"Error: Data directory {data_path} does not exist", fg=typer.colors.RED)
        typer.secho("Please run `wet-net pre-process` to download the dataset", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    results_path = Path(results_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dry_run:
        typer.secho(f"Would evaluate the {hub_model_name} model using data from {data_path}.", fg=typer.colors.YELLOW)
        typer.secho(f"Would save results to {results_path}.", fg=typer.colors.YELLOW)
        typer.secho("Rerun without the --dry-run flag to proceed.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=0)

    model = MNISTClassifierCNN().from_pretrained(hub_model_name).to(device)
    model.eval()
    typer.echo(f"Model {hub_model_name} loaded successfully.")

    test_dataset = datasets.MNIST(root=str(data_path), train=False, download=False, transform=MNIST_TRANSFORMS)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    correct = 0
    total = 0
    results = []

    typer.echo("Evaluation started...")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            total += target.size(0)
            correct += pred.eq(target).sum().item()

            for i in range(len(target)):
                results.append({"true_label": target[i].item(), "predicted_label": pred[i].item()})

    accuracy = 100.0 * correct / total
    typer.echo(f"Test Accuracy: {accuracy:.2f}%")

    df = pd.DataFrame(results)
    results_file = results_path / "evaluation_results.csv"
    results_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_file, index=False)
    typer.echo(f"Results saved to {results_file}")

    typer.secho("Evaluation completed.", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
