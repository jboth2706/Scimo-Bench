"""The model protocol: the class a model developer writes so scimo-bench can run the model.

The class lives in the developer's repo and is named in the model config
(`model="package.module:Class"`). scimo-bench creates it in a job, asks it for the task's
events, times `generate`, and writes what `to_floats` returns as a prediction file.
See `docs/PROTOCOL.md`.
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol

import awkward as ak
import numpy as np

from scimo_bench import io

METHODS = ("load", "inputs", "generate", "to_floats", "info")


class Model(Protocol):
    """The methods a model class implements (see the module docstring)."""

    def load(self, cfg: dict) -> None:
        """Build the model. `cfg` is the model config (`checkpoint`, `generation`, `options`)
        plus `runtime` = {device, precision, compile}."""

    def inputs(self, event_ids: np.ndarray, size: int) -> Any:
        """The model's inputs for these events (read from the model's own data), on the
        device. Not timed as model time."""

    def generate(self, batch: Any) -> Any:
        """Run the model on one batch. This is what is timed."""

    def to_floats(self, out: Any) -> Any:
        """`generate`'s output as one row per event: `event_id` and the task's collection,
        e.g. `particles.{pt, eta, phi, e}` as floats on the truth's scale (GeV, rad).
        An awkward array or a dict of columns."""

    def info(self) -> dict:
        """`family`, `n_params`, `generation` (the settings actually used), and optionally
        `checkpoint`, `flops_per_event`, `nfe_per_event`, `tokens_per_event`."""


def missing_methods(obj: Any) -> list[str]:
    """Protocol methods `obj` lacks."""
    return [m for m in METHODS if not callable(getattr(obj, m, None))]


def create(spec: str) -> Model:
    """Import `package.module:Class` and create the model object (not loaded yet)."""
    module, _, cls = spec.partition(":")
    if not cls:
        raise ValueError(f"model must be 'package.module:Class', got {spec!r}")
    obj = getattr(importlib.import_module(module), cls)()
    missing = missing_methods(obj)
    if missing:
        raise TypeError(f"{spec} does not implement the model protocol: missing {missing}")
    return obj


def as_array(out: Any) -> ak.Array:
    """What `to_floats` returned, in the internal layout (`io.normalize`)."""
    arr = out if isinstance(out, ak.Array) else ak.Array(out)
    return io.normalize(arr)
