"""Cutflow configs (group `cutflow`); steps run in order."""

from hydra_zen import builds, make_config, store

from scimo_bench.cutflow import CutStep

cutflow_store = store(group="cutflow")

cutflow_store(
    make_config(
        name="central",
        steps=[
            builds(CutStep, cut="max_abs_eta", value=2.5),
            builds(CutStep, cut="min_pt", value=1.0),
            builds(CutStep, cut="min_particles", value=1),
        ],
    ),
    name="central",
)
