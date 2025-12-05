from pathlib import Path

import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader
from torchvision import datasets

from wet_net.mnist_demo.classifier import MNISTClassifierCNN
from wet_net.mnist_demo.transforms import MNIST_TRANSFORMS

app = typer.Typer()

MODEL_NAME = "zachparent/mnist-classifier"


@app.command()
def evaluate(
    data_dir: str = typer.Option("./data", help="Directory containing the dataset"),
    results_dir: str = typer.Option("./results", help="Directory to save evaluation results"),
    dry_run: bool = typer.Option(False, help="Dry run the evaluation."),
):
    """
    Evaluate the model from HuggingFace and save results to CSV.
    """
    data_path = Path(data_dir)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    typer.echo("Evaluation started...")

    if not dry_run:
        model = MNISTClassifierCNN().from_pretrained(MODEL_NAME).to(device)
        model.eval()

        test_dataset = datasets.MNIST(root=str(data_path), train=False, download=False, transform=MNIST_TRANSFORMS)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        correct = 0
        total = 0
        results = []

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
        df.to_csv(results_file, index=False)
        typer.echo(f"Results saved to {results_file}")

    typer.echo("Evaluation completed.")


if __name__ == "__main__":
    app()
