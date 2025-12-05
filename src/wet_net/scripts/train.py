from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import typer
from torch.utils.data import DataLoader
from torchvision import datasets

from wet_net.mnist_demo.classifier import MNISTClassifierCNN
from wet_net.mnist_demo.transforms import MNIST_TRANSFORMS

app = typer.Typer()

MODEL_NAME = "zachparent/mnist-classifier"


@app.command()
def train(
    data_dir: str = typer.Option("./data", help="Directory containing the dataset"),
    dry_run: bool = typer.Option(False, help="Dry run the training."),
):
    """
    Train a simple CNN on MNIST and upload to HuggingFace.
    """
    data_path = Path(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    typer.echo("Training started...")

    if not dry_run:
        train_dataset = datasets.MNIST(root=str(data_path), train=True, download=False, transform=MNIST_TRANSFORMS)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        model = MNISTClassifierCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        epochs = 3
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if batch_idx % 200 == 0:
                    typer.echo(
                        f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {running_loss / (batch_idx + 1):.4f}"
                    )

        typer.echo("Training completed. Uploading to HuggingFace...")

        model.push_to_hub(MODEL_NAME)


if __name__ == "__main__":
    app()
