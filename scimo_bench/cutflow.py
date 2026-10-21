"""Cuts on the joined data, built on collection variables from `variables.py`.

Object-level cuts apply to truth and predictions alike; event-level cuts are decided on the
truth only, so every model keeps the same event set (invariant 5).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace

import awkward as ak
import numpy as np
import pandas as pd

from scimo_bench import variables
from scimo_bench.io import Joined, cutflow_row, n_objects

log = logging.getLogger(__name__)

CUTFLOW_COLUMNS = [
    "step",
    "level",
    "model",
    "collection",
    "events",
    "objects",
    "dropped",
    "missing_events",
]


@dataclass(frozen=True)
class Cut:
    """A registered cut: a test on a collection variable at object or event level."""

    name: str
    level: str
    variable: str
    fn: Callable


@dataclass
class CutStep:
    """One configured step of a cutflow."""

    cut: str
    value: float
    collection: str = "particles"


REGISTRY: dict[str, Cut] = {}


def cut(name: str, level: str, variable: str):
    """Decorator registering a cut function `(values, value) -> mask` on `variable`."""

    def wrap(fn):
        REGISTRY[name] = Cut(name, level, variable, fn)
        return fn

    return wrap


@cut("min_pt", level="object", variable="pt")
def _min_pt(values, value):
    """Keep objects with pT above the value."""
    return values > value


@cut("max_abs_eta", level="object", variable="abs_eta")
def _max_abs_eta(values, value):
    """Keep objects with |η| below the value."""
    return values < value


@cut("min_particles", level="event", variable="n_objects")
def _min_particles(values, value):
    """Keep events with at least this many objects (decided on truth)."""
    return values >= value


def check(steps: list[CutStep]) -> None:
    """Unknown cuts or variables are configuration errors."""
    for s in steps:
        if s.cut not in REGISTRY:
            raise ValueError(f"unknown cut '{s.cut}'; known: {sorted(REGISTRY)}")
        variables.get(REGISTRY[s.cut].variable)


def object_mask(c: Cut, step: CutStep, arr: ak.Array):
    """Object mask of one cut step for one array."""
    ctx = variables.Context(collection=step.collection)
    objs = variables.objects(arr, ctx)
    return c.fn(variables.get(c.variable)(objs, ctx=ctx), step.value)


def apply(joined: Joined, steps: list[CutStep]) -> Joined:
    """Apply the cutflow; one cutflow row per step and per truth/model."""
    check(steps)
    truth, preds, rows, event_ids = (
        joined.truth,
        dict(joined.preds),
        dict(joined.rows),
        joined.event_ids,
    )
    cf = list(joined.cutflow)
    for step in steps:
        c = REGISTRY[step.cut]
        coll = step.collection
        if c.level == "object":
            n_before = n_objects(truth, coll)
            truth = ak.with_field(truth, truth[coll][object_mask(c, step, truth)], coll)
            cf.append(
                cutflow_row(
                    step.cut,
                    "truth",
                    coll,
                    len(truth),
                    n_objects(truth, coll),
                    n_before - n_objects(truth, coll),
                    "object",
                )
            )
            for model, arr in preds.items():
                n_before = n_objects(arr, coll)
                try:
                    arr = ak.with_field(arr, arr[coll][object_mask(c, step, arr)], coll)
                except variables.MissingInput as e:
                    log.warning("%s: cut %s not applied (%s)", model, step.cut, e)
                preds[model] = arr
                n_after = n_objects(arr, coll)
                dropped = 0 if n_before is None else n_before - n_after
                cf.append(cutflow_row(step.cut, model, coll, len(arr), n_after, dropped, "object"))
        else:
            ctx = variables.Context(collection=coll)
            keep = np.asarray(c.fn(variables.get(c.variable)(truth[coll], ctx=ctx), step.value))
            new_index = np.cumsum(keep) - 1
            truth, event_ids = truth[keep], event_ids[keep]
            cf.append(
                cutflow_row(
                    step.cut, "truth", coll, len(truth), n_objects(truth, coll), (~keep).sum()
                )
            )
            for model, arr in preds.items():
                row_keep = keep[rows[model]]
                preds[model] = arr[row_keep]
                rows[model] = new_index[rows[model][row_keep]]
                cf.append(
                    cutflow_row(
                        step.cut,
                        model,
                        coll,
                        len(preds[model]),
                        n_objects(preds[model], coll),
                        (~row_keep).sum(),
                    )
                )
    return replace(joined, event_ids=event_ids, truth=truth, preds=preds, rows=rows, cutflow=cf)


def to_frame(rows: list[dict]) -> pd.DataFrame:
    """Cutflow rows → the `cutflow.csv` table."""
    df = pd.DataFrame(rows, columns=CUTFLOW_COLUMNS)
    for col in ("objects", "missing_events"):
        df[col] = df[col].astype("Int64")
    return df
