import os

import typer

from wet_net.data.preprocess import prepare_dataset

app = typer.Typer()


@app.command()
def pre_process(
    mock: bool = typer.Option(False, "--mock", help="Use bundled synthetic data instead of real dataset."),
    data_url: str | None = typer.Option(
        None, help="HTTPS link to the real data (parquet or zip containing parquet). Required unless --mock."
    ),
):
    """
    Download (or generate) the dataset and run preprocessing.
    """
    url = data_url or os.getenv("WETNET_DATA_URL")
    try:
        out_path = prepare_dataset(mock=mock, data_url=url)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    typer.secho(f"Preprocessed parquet ready at {out_path}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
