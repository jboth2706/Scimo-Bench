"""Runtime configs (group `runtime`): how model code runs (steps `predict`, `perf`).
The hardware itself comes from the site's job resources."""

from hydra_zen import make_config, store

runtime_store = store(group="runtime")


def runtime_conf(name, device, precision, compile, batch_size):
    """A runtime config: how model code runs."""
    return make_config(
        name=name, device=device, precision=precision, compile=compile, batch_size=batch_size
    )


runtime_store(runtime_conf("gpu", "cuda", "bf16", True, 256), name="gpu")
runtime_store(runtime_conf("cpu", "cpu", "fp32", False, 64), name="cpu")
