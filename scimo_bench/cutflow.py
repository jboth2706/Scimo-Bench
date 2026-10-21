"""Cuts on the joined data: plain functions on a collection, applied in order.

Every cut step says which side it is applied to (`on`):
- `truth`: the truth objects / events only;
- `reco`: the models' output objects only;
- `both`: truth and reco, each with its own values (a reco particle is cut on its
  reconstructed pT, a truth particle on its true pT).

Object-level cuts (`min_pt`, `max_abs_eta`) may use any of the three. Event-level cuts
(`min_particles`) remove whole events and must be `on: truth`: every model is scored on the
same events (paired comparison), so an event cannot be kept for one model and dropped for
another.

A new cut: a function `(objects, value) -> keep mask` decorated with `@cut(level)`;
`objects` is the step's collection (e.g. `objects.pt`), the mask is per object
(`level="object"`) or per event (`level="event"`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace

import awkward as ak
import numpy as np
import pandas as pd

from scimo_bench.io import Joined, cutflow_row, n_objects

log = logging.getLogger(__name__)

SIDES = ("truth", "reco", "both")
CUTFLOW_COLUMNS = [
    "step",
    "level",
    "on",
    "model",
    "collection",
    "events",
    "objects",
    "dropped",
    "missing_events",
]


@dataclass(frozen=True)
class Cut:
    """A registered cut: a test on a collection, at object or event level."""

    name: str
    level: str
    fn: Callable


@dataclass
class CutStep:
    """One configured step of a cutflow: cut `cut` with threshold `value`, applied `on`
    truth, reco (the models' output) or both, to `collection`."""

    cut: str
    value: float
    on: str
    collection: str = "particles"


REGISTRY: dict[str, Cut] = {}


def cut(level: str):
    """Decorator registering a cut function `(objects, value) -> keep mask` by its name."""

    def wrap(fn):
        REGISTRY[fn.__name__] = Cut(fn.__name__, level, fn)
        return fn

    return wrap


@cut(level="object")
def min_pt(objs, value):
    """Keep objects with pT above the value."""
    return objs.pt > value


@cut(level="object")
def max_abs_eta(objs, value):
    """Keep objects with |η| below the value."""
    return abs(objs.eta) < value


@cut(level="event")
def min_particles(objs, value):
    """Keep events with at least this many objects."""
    return ak.num(objs, axis=1) >= value


def check(steps: list[CutStep]) -> None:
    """Unknown cuts or sides, and event cuts not on truth, are config errors."""
    for s in steps:
        if s.cut not in REGISTRY:
            raise ValueError(f"unknown cut '{s.cut}'; known: {sorted(REGISTRY)}")
        if s.on not in SIDES:
            raise ValueError(f"cut {s.cut}: on must be one of {SIDES}, got {s.on!r}")
        if REGISTRY[s.cut].level == "event" and s.on != "truth":
            raise ValueError(
                f"cut {s.cut} removes events and must be on: truth (all models are scored "
                f"on the same events), got on: {s.on}"
            )


def cut_objects(c: Cut, step: CutStep, arr: ak.Array) -> ak.Array:
    """`arr` with the step's collection reduced to the objects the cut keeps."""
    objs = arr[step.collection]
    return ak.with_field(arr, objs[c.fn(objs, step.value)], step.collection)


def apply(joined: Joined, steps: list[CutStep]) -> Joined:
    """Apply the cutflow in order; one cutflow row per step and per truth/model."""
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
            if step.on in ("truth", "both"):
                truth = cut_objects(c, step, truth)
            n_after = n_objects(truth, coll)
            cf.append(
                cutflow_row(
                    step.cut,
                    "truth",
                    coll,
                    len(truth),
                    n_after,
                    n_before - n_after,
                    "object",
                    on=step.on,
                )
            )
            for model, arr in preds.items():
                n_before = n_objects(arr, coll)
                if step.on in ("reco", "both"):
                    try:
                        arr = cut_objects(c, step, arr)
                    except (AttributeError, ak.errors.FieldNotFoundError) as e:
                        log.warning("%s: cut %s not applied (%s)", model, step.cut, e)
                preds[model] = arr
                n_after = n_objects(arr, coll)
                dropped = 0 if n_before is None else n_before - n_after
                cf.append(
                    cutflow_row(
                        step.cut, model, coll, len(arr), n_after, dropped, "object", on=step.on
                    )
                )
        else:
            keep = ak.to_numpy(c.fn(truth[coll], step.value))
            new_index = np.cumsum(keep) - 1
            truth, event_ids = truth[keep], event_ids[keep]
            cf.append(
                cutflow_row(
                    step.cut,
                    "truth",
                    coll,
                    len(truth),
                    n_objects(truth, coll),
                    (~keep).sum(),
                    on=step.on,
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
                        on=step.on,
                    )
                )
    return replace(joined, event_ids=event_ids, truth=truth, preds=preds, rows=rows, cutflow=cf)


def to_frame(rows: list[dict]) -> pd.DataFrame:
    """Cutflow rows → the `cutflow.csv` table."""
    df = pd.DataFrame(rows, columns=CUTFLOW_COLUMNS)
    for col in ("objects", "missing_events"):
        df[col] = df[col].astype("Int64")
    return df
