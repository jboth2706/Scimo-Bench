"""Perf configs (group `perf`): scenarios, batch sizes, timing repeats. Device, precision
and compile come from `runtime`."""

from hydra_zen import make_config, store

perf_store = store(group="perf")


def perf_conf(name, scenarios, batch_sizes, **kw):
    """A perf config: scenarios, batch sizes and timing settings."""
    base = dict(warmup=20, repeats=5, n_events=20000, single_events=1000, n_gpus=1)
    return make_config(name=name, scenarios=scenarios, batch_sizes=batch_sizes, **{**base, **kw})


perf_store(perf_conf("single", ["single"], [1]), name="single")
perf_store(perf_conf("offline", ["offline"], [1, 64, 256]), name="offline")
perf_store(perf_conf("both", ["single", "offline"], [1, 64, 256]), name="both")
# code-path check (tests, CPU); not a measurement
perf_store(
    perf_conf(
        "smoke", ["single", "offline"], [1, 8], warmup=2, repeats=3, n_events=64, single_events=16
    ),
    name="smoke",
)
