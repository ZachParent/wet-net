from pathlib import Path

import typer
from torchvision import datasets

from wet_net.mnist_demo.transforms import MNIST_TRANSFORMS

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

    typer.echo(f"Downloading MNIST dataset to {data_path}...")

    if dry_run:
        typer.secho(f"Would download MNIST the dataset to {data_path}.", fg=typer.colors.YELLOW)
        typer.secho("Rerun without the --dry-run flag to proceed.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=0)

    data_path.mkdir(parents=True, exist_ok=True)
    datasets.MNIST(root=str(data_path), train=True, download=True, transform=MNIST_TRANSFORMS)
    datasets.MNIST(root=str(data_path), train=False, download=True, transform=MNIST_TRANSFORMS)

    typer.secho(f"MNIST dataset downloaded successfully to {data_path}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
