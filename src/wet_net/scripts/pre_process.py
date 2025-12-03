import time

import typer

app = typer.Typer()


@app.command()
def pre_process(dry_run: bool = typer.Option(False, help="Dry run the pre-processing.")):
    """
    Pre-process data (dummy implementation).
    """
    typer.echo("Pre-processing started...")
    if not dry_run:
        with typer.progressbar(range(100), label="Pre-processing") as progress:
            for _ in progress:
                time.sleep(0.01)
    typer.echo("Pre-processing completed.")


if __name__ == "__main__":
    app()
