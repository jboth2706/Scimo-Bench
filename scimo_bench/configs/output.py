"""Output configs (group `output`): the run folder (`<paths.outputs>/<run>`), plot file
formats, bootstrap settings, seed."""

from hydra_zen import make_config, store

output_store = store(group="output")

output_store(
    make_config(
        dir="${paths.outputs}/${run}",
        formats=["pdf"],  # e.g. output.formats=[pdf,png]
        bootstrap=dict(n=1000, level=0.95),
        on_missing="warn",
        seed=0,
    ),
    name="default",
)
