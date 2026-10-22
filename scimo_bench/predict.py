"""The predict step: run each model's protocol class over the task's events and write the
prediction file (contract v1) that the quality path reads.

One failing model does not stop the others: its old prediction file is removed first, so
scoring reports it as `missing` instead of scoring stale predictions.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
from collections.abc import Mapping
from pathlib import Path

import awkward as ak
import numpy as np

from scimo_bench import io, jobs, perf, protocol
from scimo_bench.schema import EVENT_ID, META_FILE, SCHEMA_VERSION, Meta, condition_hash, validate

log = logging.getLogger(__name__)


def check_generation(name: str, configured: Mapping, reported: Mapping | None) -> None:
    """The model's reported `generation` must equal the configured one."""
    if reported is not None and condition_hash(dict(reported)) != condition_hash(dict(configured)):
        raise ValueError(
            f"{name}: the model reports generation {dict(reported)}, but the config says "
            f"{dict(configured)}; they must match"
        )


def clear(path: Path) -> None:
    """Remove a previous prediction file of this model (only contract files)."""
    if not path.is_dir():
        return
    for f in path.glob("*.parquet"):
        f.unlink()
    (path / META_FILE).unlink(missing_ok=True)


def generate(model, rt: perf.Runtime, events: np.ndarray) -> tuple[ak.Array, float]:
    """All events through inputs → generate → to_floats; (predictions, model seconds)."""
    parts, model_s = [], 0.0
    for i in range(0, len(events), rt.batch_size):
        ids = events[i : i + rt.batch_size]
        out, dt_s = perf.timed(model.generate, rt.device, model.inputs(ids, len(ids)))
        model_s += dt_s
        parts.append(protocol.as_array(model.to_floats(out)))
    return ak.concatenate(parts), model_s


def to_float32(arr: ak.Array) -> ak.Array:
    """Collection fields as float32 (contract v1); the class `cls` as int32."""
    for coll in arr.fields:
        if arr[coll].fields:
            sub = {
                f: ak.values_astype(arr[coll][f], np.int32 if f == "cls" else np.float32)
                for f in arr[coll].fields
            }
            arr = ak.with_field(arr, ak.zip(sub), coll)
    return arr


def event_list_hash(events: np.ndarray) -> str:
    """Short hash of the event ids (recorded in `meta`)."""
    return hashlib.sha1(np.asarray(events, dtype=np.int64).tobytes()).hexdigest()[:10]


def run_model(
    model_cfg: Mapping, rt: perf.Runtime, events: np.ndarray, task: Mapping, dataset: Mapping
) -> tuple[protocol.Model, float, dict]:
    """Load the model, predict all `events`, write its prediction file (contract v1, with
    `meta`); returns (model, load seconds, info) so perf can reuse the loaded model."""
    name = model_cfg["name"]
    out = io.resolve_path(model_cfg["predictions"])
    clear(out)
    loaded = perf.load_model(model_cfg, rt)
    model, load_s, info = loaded
    generation = dict(model_cfg.get("generation") or {})
    check_generation(name, generation, info.get("generation"))
    t0 = dt.datetime.now()
    arr, model_s = generate(model, rt, events)
    wall_s = (dt.datetime.now() - t0).total_seconds()
    arr = to_float32(arr)
    got = set(ak.to_numpy(arr[EVENT_ID]).tolist())
    lost = len(set(events.tolist()) - got)
    if lost:
        log.warning("%s: no prediction for %d of %d events", name, lost, len(events))
    meta = Meta(
        schema_version=SCHEMA_VERSION,
        model=dict(
            name=name,
            family=info.get("family") or model_cfg.get("family"),
            checkpoint=info.get("checkpoint") or model_cfg.get("checkpoint"),
            n_params=info.get("n_params"),
        ),
        generation=generation,
        data=dict(
            task=task["name"],
            dataset=dataset["name"],
            source=dataset["truth"].get("path"),
            n_events=len(events),
            event_list=task.get("events"),
            event_list_hash=event_list_hash(events),
        ),
        code=dict(
            producer_repo="scimo_bench",
            producer=model_cfg["model"],
            scimo_bench_commit=jobs.git_commit(),
        ),
        hardware=perf.hardware(rt),
        created=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        extra=dict(load_s=load_s, model_s=model_s, wall_s=wall_s, batch_size=rt.batch_size),
    )
    problems = validate(arr, meta, dataset)
    for p in problems:
        log.warning("%s: %s", name, p)
    io.write_predictions(out, arr, meta)
    log.info(
        "%s: %d events predicted in %.1f s (model %.1f s, load %.1f s) -> %s",
        name,
        len(arr),
        wall_s,
        model_s,
        load_s,
        out,
    )
    return loaded
