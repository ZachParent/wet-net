import typer

from wet_net.scripts.pre_process import pre_process
from wet_net.scripts.train import train

app = typer.Typer()

app.command(name="pre-process")(pre_process)
app.command(name="train")(train)

if __name__ == "__main__":
    app()
