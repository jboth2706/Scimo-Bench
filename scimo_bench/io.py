"""Read prediction files (contract v1), clean them, join them with the truth on `event_id`.
Truth readers live with their dataset (`scimo_bench/datasets/`).

Internal layout of every array: `event_id` (int64), `sample_id` (int64, predictions only),
one field per collection as a jagged record array (`particles: var * {pt, eta, ...}`) and
scalar event fields. `raw.*` is dropped on read.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow.parquet as pq

from scimo_bench.schema import EVENT_ID, META_FILE, META_KEY, RAW_PREFIX, SAMPLE_ID, Meta

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Joined:
    """Truth and predictions on the common event set.

    `truth[i]` is event `event_ids[i]`; prediction row `j` of a model belongs to event
    `rows[model][j]` (several rows per event when a model has several samples).
    """

    event_ids: np.ndarray
    truth: ak.Array
    preds: dict[str, ak.Array]
    rows: dict[str, np.ndarray]
    cutflow: list[dict] = field(default_factory=list)

    @property
    def models(self) -> list[str]:
        """Model names, in order."""
        return list(self.preds)

    def truth_rows(self, model: str) -> ak.Array:
        """Truth rows aligned one to one with this model's prediction rows."""
        return self.truth[self.rows[model]]


def cutflow_row(
    step, model, collection, events, objects, dropped=0, level="event", missing_events=None
) -> dict:
    """One row of `cutflow.csv`."""
    return dict(
        step=step,
        level=level,
        model=model,
        collection=collection,
        events=int(events),
        objects=None if objects is None else int(objects),
        dropped=int(dropped),
        missing_events=missing_events,
    )


def resolve_path(path: str | os.PathLike) -> Path:
    """Absolute paths as given; relative ones against the repo root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def read_event_list(path: str | os.PathLike) -> np.ndarray:
    """Event ids from a text file (one per line; relative to the repo)."""
    return np.loadtxt(resolve_path(path), dtype=np.int64, ndmin=1)


# --- predictions ---------------------------------------------------------------------------


def parquet_files(path: str | os.PathLike) -> list[str]:
    """The Parquet file(s) of a prediction path (file, or `part-*.parquet` in a folder)."""
    p = Path(path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        files = sorted(glob.glob(str(p / "part-*.parquet"))) or sorted(
            glob.glob(str(p / "*.parquet"))
        )
        if files:
            return files
    raise FileNotFoundError(f"no prediction Parquet at {p}")


def read_meta(path: str | os.PathLike, first_file: str | None = None) -> Meta | None:
    """`meta.json` next to the file(s), else Parquet key-value metadata `scimo_bench.meta`."""
    p = Path(path)
    sidecar = (p if p.is_dir() else p.parent) / META_FILE
    if sidecar.exists():
        return Meta.from_dict(json.loads(sidecar.read_text()))
    if first_file is not None:
        kv = pq.read_schema(first_file).metadata or {}
        if META_KEY.encode() in kv:
            return Meta.from_dict(json.loads(kv[META_KEY.encode()]))
    return None


def normalize(arr: ak.Array) -> ak.Array:
    """Bring any contract-v1 layout into the internal layout.

    Accepts flat dotted columns (`particles.pt`), a record of lists or a list of records.
    """
    collections: dict[str, dict[str, ak.Array]] = {}
    out: dict[str, ak.Array] = {}
    for name in arr.fields:
        if name == RAW_PREFIX or name.startswith(RAW_PREFIX + "."):
            continue
        if "." in name:
            coll, sub = name.split(".", 1)
            collections.setdefault(coll, {})[sub] = arr[name]
        elif arr[name].fields:
            for sub in arr[name].fields:
                collections.setdefault(name, {})[sub] = arr[name][sub]
        else:
            out[name] = arr[name]
    if EVENT_ID in out:
        out[EVENT_ID] = ak.values_astype(out[EVENT_ID], np.int64)
    if SAMPLE_ID in out:
        out[SAMPLE_ID] = ak.values_astype(out[SAMPLE_ID], np.int64)
    elif EVENT_ID in out:
        out[SAMPLE_ID] = ak.Array(np.zeros(len(arr), dtype=np.int64))
    for coll, subs in collections.items():
        counts = {k: ak.to_numpy(ak.num(v, axis=1)) for k, v in subs.items()}
        ref = next(iter(counts.values()))
        bad = [k for k, c in counts.items() if not np.array_equal(c, ref)]
        if bad:
            raise ValueError(f"collection '{coll}': fields {bad} differ in length from the rest")
        out[coll] = ak.zip(subs, depth_limit=None)
    return ak.zip(out, depth_limit=1)


def read_predictions(path: str | os.PathLike) -> tuple[ak.Array, Meta | None]:
    """A contract file or a directory of `part-*.parquet` → (normalized array, meta)."""
    files = parquet_files(path)
    parts = [ak.from_parquet(f) for f in files]
    arr = parts[0] if len(parts) == 1 else ak.concatenate(parts)
    return normalize(arr), read_meta(path, files[0])


def write_predictions(directory: str | os.PathLike, arr: ak.Array, meta: Meta | dict) -> Path:
    """Write a contract-v1 directory: `part-000.parquet` + `meta.json`."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "part-000.parquet"
    ak.to_parquet(arr, path)
    meta_d = meta.to_dict() if isinstance(meta, Meta) else dict(meta)
    (d / META_FILE).write_text(json.dumps(meta_d, indent=2, default=str))
    return path


# --- cleaning ------------------------------------------------------------------------------


def collections_of(arr: ak.Array) -> list[str]:
    """Names of the jagged record fields (collections) of an array."""
    return [f for f in arr.fields if arr[f].fields and f not in (EVENT_ID, SAMPLE_ID)]


def duplicate_groups(arr: ak.Array, collection: str = "particles"):
    """(keep mask over flat objects, sizes of groups of identical (pt, eta, phi) objects)."""
    objs = arr[collection]
    c = ak.to_numpy(ak.num(objs, axis=1))
    evt = np.repeat(np.arange(len(c)), c)
    key = np.stack(
        [evt.astype(np.float64)]
        + [ak.to_numpy(ak.flatten(objs[k])).astype(np.float64) for k in ("pt", "eta", "phi")],
        axis=1,
    )
    keep = np.zeros(len(key), dtype=bool)
    if len(key) == 0:
        return keep, np.zeros(0, dtype=np.int64)
    _, first, sizes = np.unique(key, axis=0, return_index=True, return_counts=True)
    keep[first] = True
    return keep, sizes[sizes > 1]


def dedupe_objects(arr: ak.Array, collection: str = "particles") -> tuple[ak.Array, int]:
    """Drop objects identical in (pt, eta, phi) to an earlier one in the same event.

    Returns the number dropped; groups of k identical objects count k - 1 (and k(k-1)/2 pairs).
    """
    keep, groups = duplicate_groups(arr, collection)
    n_dropped = int((~keep).sum())
    if n_dropped:
        log.info(
            "duplicates in %s: %d pairs in %d groups, %d objects dropped",
            collection,
            int((groups * (groups - 1) // 2).sum()),
            len(groups),
            n_dropped,
        )
        objs = arr[collection]
        arr = ak.with_field(arr, objs[ak.unflatten(keep, ak.num(objs, axis=1))], collection)
    return arr, n_dropped


def drop_nonfinite(arr: ak.Array) -> tuple[ak.Array, int]:
    """Drop objects with any non-finite float field; return the number dropped."""
    total = 0
    for coll in collections_of(arr):
        objs = arr[coll]
        ok = None
        for f in objs.fields:
            vals = objs[f]
            if not np.issubdtype(ak.to_numpy(ak.flatten(vals)).dtype, np.floating):
                continue
            fin = np.isfinite(vals)
            ok = fin if ok is None else (ok & fin)
        if ok is None:
            continue
        n = int(ak.sum(~ok))
        if n:
            arr = ak.with_field(arr, objs[ok], coll)
            total += n
    return arr, total


def n_objects(arr: ak.Array, collection: str) -> int | None:
    """Total objects of a collection (None if the array lacks it)."""
    return int(ak.sum(ak.num(arr[collection], axis=1))) if collection in arr.fields else None


# --- join ----------------------------------------------------------------------------------


def join(
    truth: ak.Array, preds_by_model: dict[str, ak.Array], collection: str = "particles"
) -> Joined:
    """Inner join on `event_id`; the event set is the intersection over all models."""
    rows_cf: list[dict] = []
    truth_ids = ak.to_numpy(truth[EVENT_ID])
    rows_cf.append(
        cutflow_row("input", "truth", collection, len(truth), n_objects(truth, collection))
    )
    truth, n_dup = dedupe_objects(truth, collection)
    rows_cf.append(
        cutflow_row(
            "dedupe", "truth", collection, len(truth), n_objects(truth, collection), n_dup, "object"
        )
    )

    cleaned: dict[str, ak.Array] = {}
    common = np.unique(truth_ids)
    for model, arr in preds_by_model.items():
        rows_cf.append(
            cutflow_row("input", model, collection, len(arr), n_objects(arr, collection))
        )
        arr, n_bad = drop_nonfinite(arr)
        if n_bad:
            log.warning("%s: dropped %d objects with non-finite values", model, n_bad)
        rows_cf.append(
            cutflow_row(
                "finite", model, collection, len(arr), n_objects(arr, collection), n_bad, "object"
            )
        )
        ids = ak.to_numpy(arr[EVENT_ID])
        known = np.isin(ids, truth_ids)
        if (~known).any():
            log.warning(
                "%s: %d rows with event_id not in the truth, dropped", model, (~known).sum()
            )
            arr = arr[known]
        rows_cf.append(
            cutflow_row(
                "known_ids", model, collection, len(arr), n_objects(arr, collection), (~known).sum()
            )
        )
        keys = np.stack([ak.to_numpy(arr[EVENT_ID]), ak.to_numpy(arr[SAMPLE_ID])], axis=1)
        _, first = np.unique(keys, axis=0, return_index=True)
        if len(first) < len(arr):
            log.warning(
                "%s: %d duplicate (event_id, sample_id) rows, dropped", model, len(arr) - len(first)
            )
            arr = arr[np.sort(first)]
        cleaned[model] = arr
        common = np.intersect1d(common, ak.to_numpy(arr[EVENT_ID]))

    t_idx = np.searchsorted(truth_ids, common, sorter=np.argsort(truth_ids, kind="stable"))
    t_order = np.argsort(truth_ids, kind="stable")[t_idx]
    joined_truth = truth[t_order]
    lost = len(truth_ids) - len(common)
    if lost:
        log.warning("join: %d of %d truth events not in every model", lost, len(truth_ids))
    rows_cf.append(
        cutflow_row(
            "join", "truth", collection, len(common), n_objects(joined_truth, collection), lost
        )
    )

    preds: dict[str, ak.Array] = {}
    rows: dict[str, np.ndarray] = {}
    for model, arr in cleaned.items():
        ids = ak.to_numpy(arr[EVENT_ID])
        missing = len(np.setdiff1d(truth_ids, ids))
        if missing:
            log.warning("%s: %d truth events have no prediction", model, missing)
        keep = np.isin(ids, common)
        arr = arr[keep]
        order = np.lexsort((ak.to_numpy(arr[SAMPLE_ID]), ak.to_numpy(arr[EVENT_ID])))
        arr = arr[order]
        preds[model] = arr
        rows[model] = np.searchsorted(common, ak.to_numpy(arr[EVENT_ID]))
        n_obj = n_objects(arr, collection)
        rows_cf.append(
            cutflow_row("join", model, collection, len(arr), n_obj, (~keep).sum(), "event", missing)
        )
    return Joined(common, joined_truth, preds, rows, rows_cf)
