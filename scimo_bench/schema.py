"""Prediction contract v1 (docs/CONTRACT.md): field names, types, `Meta`, validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import awkward as ak
import numpy as np

SCHEMA_VERSION = 1
SUPPORTED_VERSIONS = (1,)

EVENT_ID = "event_id"
SAMPLE_ID = "sample_id"
META_FILE = "meta.json"
META_KEY = "scimo_bench.meta"
RAW_PREFIX = "raw"

FAMILIES = ("token_ar", "masked", "discrete_diffusion", "continuous", "regression")
MULTIPLICITIES = ("model", "oracle")


@dataclass
class Meta:
    """The `meta` block of a prediction file."""

    schema_version: int = SCHEMA_VERSION
    model: dict = field(default_factory=dict)
    generation: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    code: dict = field(default_factory=dict)
    hardware: dict = field(default_factory=dict)
    created: str | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict | None) -> Meta:
        """From a `meta.json` dict (unknown keys go to `extra`)."""
        d = dict(d or {})
        known = {k: d.pop(k) for k in list(d) if k in cls.__dataclass_fields__ and k != "extra"}
        return cls(**known, extra=d)

    def to_dict(self) -> dict:
        """As a `meta.json` dict."""
        d = asdict(self)
        d.update(d.pop("extra"))
        return d

    @property
    def condition(self) -> str:
        """Hash of the generation block (the report's `condition`)."""
        return condition_hash(self.generation)


def condition_hash(generation: dict | None) -> str:
    """Short stable hash of a generation config; the report's quality `condition`.

    Keys with value `None` are ignored, so an omitted knob and an explicit null agree.
    """
    gen = {k: v for k, v in dict(generation or {}).items() if v is not None}
    blob = json.dumps(gen, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def task_fields(task: Any) -> tuple[dict[str, list[str]], list[str]]:
    """(collections -> object fields, event fields) a task declares."""
    if task is None:
        return {}, []
    collections = {k: list(v) for k, v in dict(task.get("collections") or {}).items()}
    return collections, list(task.get("event_fields") or [])


def validate(array: ak.Array, meta: Meta | None, task: Any = None) -> list[str]:
    """Problems of a normalized prediction array against contract v1 (empty = valid)."""
    problems: list[str] = []
    fields = set(array.fields)
    if EVENT_ID not in fields:
        problems.append(f"missing required field '{EVENT_ID}'")
    else:
        ids = ak.to_numpy(array[EVENT_ID])
        if not np.issubdtype(ids.dtype, np.integer):
            problems.append(f"'{EVENT_ID}' must be integer, got {ids.dtype}")
        samples = ak.to_numpy(array[SAMPLE_ID]) if SAMPLE_ID in fields else np.zeros_like(ids)
        keys = np.stack([ids.astype(np.int64), samples.astype(np.int64)], axis=1)
        n_dup = len(keys) - len(np.unique(keys, axis=0)) if len(keys) else 0
        if n_dup:
            problems.append(f"{n_dup} duplicate ({EVENT_ID}, {SAMPLE_ID}) rows")

    collections, event_fields = task_fields(task)
    for coll, wanted in collections.items():
        if coll not in fields:
            problems.append(f"missing collection '{coll}'")
            continue
        have = set(array[coll].fields)
        problems += [f"missing field '{coll}.{f}'" for f in wanted if f not in have]
    problems += [f"missing event field '{f}'" for f in event_fields if f not in fields]

    if meta is None:
        problems.append("no meta")
        return problems
    if meta.schema_version not in SUPPORTED_VERSIONS:
        problems.append(f"unsupported schema_version {meta.schema_version}")
    family = meta.model.get("family")
    if family is not None and family not in FAMILIES:
        problems.append(f"unknown family '{family}'")
    mult = meta.generation.get("multiplicity")
    if mult is not None and mult not in MULTIPLICITIES:
        problems.append(f"unknown multiplicity '{mult}'")
    return problems
