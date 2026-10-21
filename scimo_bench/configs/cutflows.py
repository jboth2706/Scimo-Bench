"""Cutflow configs (group `cutflow`): steps run in order. Every step names its side,
`on: truth | reco | both` (reco = the models' output); event-level cuts must be
`on: truth` so that all models keep the same events. See `scimo_bench/cutflow.py`."""

from hydra_zen import builds, make_config, store

from scimo_bench.cutflow import CutStep

cutflow_store = store(group="cutflow")


def step(cut: str, value: float, on: str, collection: str = "particles"):
    """One cut step: `cut` at `value`, applied `on` truth, reco or both."""
    return builds(CutStep, cut=cut, value=value, on=on, collection=collection)


cutflow_store(
    make_config(
        name="preselection",
        steps=[
            # acceptance on both sides: each particle cut on its own |eta| / pT
            step("max_abs_eta", 2.5, on="both"),
            step("min_pt", 1.0, on="both"),
            # events with at least one true particle left (decided on truth)
            step("min_particles", 1, on="truth"),
        ],
    ),
    name="preselection",
)
