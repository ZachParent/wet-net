import typer

from wet_net.scripts.evaluate import evaluate
from wet_net.scripts.hf_util import hf_check
from wet_net.scripts.pre_process import pre_process
from wet_net.scripts.train import train
from wet_net.scripts.visualize import class_balance

app = typer.Typer()

app.command(name="pre-process")(pre_process)
app.command(name="train")(train)
app.command(name="evaluate")(evaluate)
app.command(name="hf-check")(hf_check)
app.command(name="class-balance")(class_balance)

if __name__ == "__main__":
    app()
