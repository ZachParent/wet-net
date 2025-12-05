from pathlib import Path

import typer
from torchvision import datasets, transforms

app = typer.Typer()


@app.command()
def pre_process(
    data_dir: str = typer.Option("./data", help="Directory to save the dataset"),
    dry_run: bool = typer.Option(False, help="Dry run the pre-processing."),
):
    """
    Download and preprocess MNIST dataset.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Downloading MNIST dataset to {data_path}...")

    if not dry_run:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        datasets.MNIST(root=str(data_path), train=True, download=True, transform=transform)
        datasets.MNIST(root=str(data_path), train=False, download=True, transform=transform)

        typer.echo(f"MNIST dataset downloaded successfully to {data_path}")


if __name__ == "__main__":
    app()
