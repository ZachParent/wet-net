import time

import typer

app = typer.Typer()


@app.command()
def train(dry_run: bool = typer.Option(False, help="Dry run the training.")):
    """
    Train the model (dummy implementation).
    """
    typer.echo("Training started...")
    if not dry_run:
        with typer.progressbar(range(100), label="Training") as progress:
            for _ in progress:
                time.sleep(0.01)
    typer.echo("Training completed.")


if __name__ == "__main__":
    app()
