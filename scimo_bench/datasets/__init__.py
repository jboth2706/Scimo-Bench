"""Eval datasets: how to read a dataset's truth, and its observables, comparisons, plots.

A dataset is a config entry (group `dataset`, `scimo_bench/configs/datasets.py`) naming
- `truth`: a reader object with `read(event_ids) -> ak.Array` in the internal layout
  (`event_id` + one jagged record field per collection, values in physical units);
- `module`: its package here (e.g. `scimo_bench.datasets.cocoa`) with `observables.py`,
  `comparisons.py`, `plots.py` and a `CATALOGUE` (see `scimo_bench/catalogue.py`).

Shared physics helpers (jets, matching, EMD) are in `common.py`.
A new dataset = a package here (reader + the three modules) and one entry in the config.
"""

from __future__ import annotations

import os
from typing import Protocol

import awkward as ak
import numpy as np

from scimo_bench import io
from scimo_bench.schema import EVENT_ID, SAMPLE_ID


class TruthReader(Protocol):
    """What a dataset's truth reader provides."""

    def read(self, event_ids: np.ndarray | None = None) -> ak.Array:
        """Truth of these events (all if None): `event_id` + collections."""


class ContractTruth:
    """Truth already in the contract layout (a Parquet file or folder, as predictions)."""

    def __init__(self, path: str | os.PathLike):
        self.path = path

    def read(self, event_ids: np.ndarray | None = None) -> ak.Array:
        """The file's events (only `event_ids` if given), without `sample_id`."""
        arr, _ = io.read_predictions(io.resolve_path(self.path))
        arr = arr[[f for f in arr.fields if f != SAMPLE_ID]]
        if event_ids is not None:
            arr = arr[np.isin(ak.to_numpy(arr[EVENT_ID]), np.asarray(event_ids))]
        return arr
