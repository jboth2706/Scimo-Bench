"""Output configs (group `output`): one folder per call, bootstrap settings, seed."""

from hydra_zen import make_config, store

output_store = store(group="output")

output_store(
    make_config(
        dir="${paths.outputs}/${task.name}/${now:%Y-%m-%d_%H-%M-%S}",
        formats=["png", "pdf"],
        bootstrap=dict(n=1000, level=0.95),
        on_missing="warn",
        seed=0,
    ),
    name="default",
)
