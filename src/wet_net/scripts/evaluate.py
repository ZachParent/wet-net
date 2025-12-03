import time

import typer

app = typer.Typer()


@app.command()
def evaluate(dry_run: bool = typer.Option(False, help="Dry run the evaluation.")):
    """
    Evaluate the model (dummy implementation).
    """
    typer.echo("Evaluation started...")
    if not dry_run:
        with typer.progressbar(range(100), label="Evaluation") as progress:
            for _ in progress:
                time.sleep(0.01)
    typer.echo("Evaluation completed.")


if __name__ == "__main__":
    app()
