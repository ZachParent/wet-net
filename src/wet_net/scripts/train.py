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


@app.command()
def train(
    dry_run: bool = typer.Option(False, help="Dry run the training."),
    upload_to_hub: bool = typer.Option(False, help="Upload the trained model to HuggingFace."),
    upload_only: bool = typer.Option(False, help="Upload the existing model to HuggingFace without retraining."),
    data_dir: str = typer.Option("./data", help="Directory containing the dataset"),
    local_model_path: str = typer.Option("./models/mnist-classifier.pt", help="Path to save the trained model"),
    hub_model_name: str = typer.Option(
        "zachparent/mnist-classifier", help="Name of the model to upload to HuggingFace"
    ),
):
    """
    Train a simple CNN on MNIST and optionally upload to HuggingFace.
    """
    upload_to_hub = upload_to_hub or upload_only
    data_path = Path(data_dir)
    local_model_path = Path(local_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not data_path.exists():
        typer.secho(f"Error: Data directory {data_path} does not exist", fg=typer.colors.RED)
        typer.secho("Please run `wet-net pre-process` to download the dataset", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    if dry_run:
        if upload_only:
            typer.secho(
                f"Would upload the model {local_model_path} to huggingface as {hub_model_name}.", fg=typer.colors.YELLOW
            )
            typer.secho("Rerun without the --dry-run flag to proceed.", fg=typer.colors.YELLOW)
            raise typer.Exit(code=0)
        typer.secho(f"Would train the model using data from {data_path}.", fg=typer.colors.YELLOW)
        typer.secho(f"Would save the trained model to {local_model_path}.", fg=typer.colors.YELLOW)
        if upload_to_hub:
            typer.secho(f"Would upload the model to huggingface as {hub_model_name}.", fg=typer.colors.YELLOW)
        typer.secho("Rerun without the --dry-run flag to proceed.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=0)

    if upload_only:
        typer.echo(f"Uploading model {local_model_path} to HuggingFace as {hub_model_name}...")
        model: MNISTClassifierCNN = torch.load(local_model_path, weights_only=False)
        model.push_to_hub(hub_model_name)
        typer.secho(f"Model uploaded to {hub_model_name}", fg=typer.colors.GREEN)
        raise typer.Exit(code=0)

    typer.echo("Training started...")
    train_dataset = datasets.MNIST(root=str(data_path), train=True, download=False, transform=MNIST_TRANSFORMS)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = MNISTClassifierCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 1
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
                typer.echo(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {running_loss / (batch_idx + 1):.4f}")

    typer.secho("Training completed.", fg=typer.colors.GREEN)

    local_model_path.parent.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Saving model to {local_model_path}...")
    torch.save(model, local_model_path)
    typer.secho(f"Model saved to {local_model_path}", fg=typer.colors.GREEN)
    if upload_to_hub:
        typer.echo(f"Uploading model to HuggingFace as {hub_model_name}...")
        model.push_to_hub(hub_model_name)
        typer.secho(f"Model uploaded to {hub_model_name}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
