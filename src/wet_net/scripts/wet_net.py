import typer

from wet_net.scripts.evaluate import evaluate
from wet_net.scripts.hf_util import hf_check
from wet_net.scripts.pre_process import pre_process
from wet_net.scripts.train import train

app = typer.Typer()

app.command(name="pre-process")(pre_process)
app.command(name="train")(train)
app.command(name="evaluate")(evaluate)
app.command(name="hf-check")(hf_check)

if __name__ == "__main__":
    app()
