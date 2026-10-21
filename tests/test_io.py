import json

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scimo_bench import io
from scimo_bench.datasets.cocoa.reader import CocoaTruth
from scimo_bench.schema import Meta, condition_hash, validate
from tests import synthetic
from tests.conftest import COCOA_VAL

TASK = {"collections": {"particles": ["pt", "eta", "phi", "e"]}}


def test_roundtrip(tmp_path, truth):
    pred = synthetic.smeared(truth)
    d = synthetic.write(tmp_path, "m", pred, dict(mode="smear", temperature=None))
    arr, meta = io.read_predictions(d)
    assert len(arr) == len(pred)
    assert set(arr.fields) == {"event_id", "sample_id", "particles"}
    assert arr.event_id.type.content.primitive == "int64"
    assert ak.all(arr.particles.pt == pred.particles.pt)
    assert meta.model["name"] == "m" and meta.generation["mode"] == "smear"
    assert validate(arr, meta, TASK) == []


def test_flat_dotted_columns_and_kv_meta(tmp_path, truth):
    """A producer writing `particles.pt` columns and meta in Parquet key-value metadata."""
    p = truth.particles
    table = pa.table(
        {
            "event_id": ak.to_numpy(truth.event_id),
            **{f"particles.{k}": ak.to_arrow(p[k], extensionarray=False) for k in p.fields},
            "raw.tokens": ak.to_arrow(ak.values_astype(p.pt, np.int32), extensionarray=False),
        }
    )
    meta = Meta(model=dict(name="flat", family="token_ar"), generation=dict(mode="argmax"))
    table = table.replace_schema_metadata({"scimo_bench.meta": json.dumps(meta.to_dict())})
    pq.write_table(table, tmp_path / "flat.parquet")
    arr, m = io.read_predictions(tmp_path / "flat.parquet")
    assert set(arr.fields) == {"event_id", "sample_id", "particles"}  # raw.* dropped
    assert set(arr.particles.fields) == {"pt", "eta", "phi", "e"}
    assert ak.all(arr.sample_id == 0)
    assert m.model["family"] == "token_ar"
    assert validate(arr, m, TASK) == []


def test_sharded_parts(tmp_path, truth):
    pred = synthetic.identical(truth)
    d = tmp_path / "sharded"
    d.mkdir()
    ak.to_parquet(pred[:100], d / "part-000.parquet")
    ak.to_parquet(pred[100:], d / "part-001.parquet")
    (d / "meta.json").write_text('{"schema_version": 1}')
    arr, meta = io.read_predictions(d)
    assert len(arr) == len(pred) and meta.schema_version == 1


def test_validate_reports_problems(truth):
    pred = synthetic.missing_field(synthetic.identical(truth), "e")
    no_id = pred[["sample_id", "particles"]]
    probs = validate(no_id, Meta(), TASK)
    assert any("event_id" in p for p in probs)
    probs = validate(pred, Meta(), TASK)
    assert probs == ["missing field 'particles.e'"]
    probs = validate(pred, Meta(schema_version=7, model={"family": "llm"}), {})
    assert any("schema_version" in p for p in probs) and any("family" in p for p in probs)
    assert "no meta" in validate(pred, None, {})


def test_validate_duplicate_rows(truth):
    pred = synthetic.identical(truth)
    dup = ak.concatenate([pred, pred[:3]])
    assert any("3 duplicate" in p for p in validate(dup, Meta(), TASK))


def test_mismatched_field_lengths_are_rejected(truth):
    p = truth.particles
    bad = ak.zip(
        {
            "event_id": truth.event_id,
            "particles": ak.zip({"pt": p.pt, "eta": p.eta[:, :1]}, depth_limit=1),
        },
        depth_limit=1,
    )
    with pytest.raises(ValueError, match="differ in length"):
        io.normalize(bad)


def test_condition_hash_ignores_nulls():
    assert condition_hash({"mode": "argmax", "temperature": None}) == condition_hash(
        {"mode": "argmax"}
    )
    assert condition_hash({"mode": "argmax"}) != condition_hash({"mode": "sample"})


def test_join_missing_events(truth):
    pred_a, dropped_ids = synthetic.missing_events(synthetic.identical(truth), n=10)
    pred_b = synthetic.identical(truth)
    j = io.join(truth, {"a": pred_a, "b": pred_b})
    assert len(j.event_ids) == len(truth) - 10
    assert not np.isin(dropped_ids, j.event_ids).any()
    assert len(j.preds["a"]) == len(j.preds["b"]) == len(j.event_ids)
    assert np.array_equal(j.rows["a"], np.arange(len(j.event_ids)))
    rows = {(r["step"], r["model"]): r for r in j.cutflow}
    assert rows[("join", "a")]["missing_events"] == 10
    assert rows[("join", "b")]["missing_events"] == 0
    assert rows[("join", "b")]["dropped"] == 10  # dropped because model a lacks them
    assert rows[("join", "truth")]["dropped"] == 10
    # truth rows aligned to model rows
    assert ak.all(j.truth_rows("b").event_id == j.preds["b"].event_id)


def test_join_unknown_ids_and_nonfinite(truth):
    pred = synthetic.identical(truth)
    alien = ak.with_field(pred[:5], pred[:5].event_id + 10**8, "event_id")
    pt = ak.to_numpy(ak.flatten(pred.particles.pt)).copy()
    pt[[0, 3]] = np.nan
    bad = ak.with_field(pred, ak.unflatten(pt, ak.num(pred.particles)), ("particles", "pt"))
    j = io.join(truth, {"m": ak.concatenate([bad, alien])})
    rows = {(r["step"], r["model"]): r for r in j.cutflow}
    assert rows[("known_ids", "m")]["dropped"] == 5
    assert rows[("finite", "m")]["dropped"] == 2
    assert len(j.event_ids) == len(truth)


def test_join_several_samples(truth):
    a = synthetic.identical(truth)
    b = ak.with_field(synthetic.smeared(truth), np.ones(len(truth), dtype=np.int64), "sample_id")
    pred = ak.concatenate([a, b])
    j = io.join(truth, {"m": io.normalize(pred)})
    assert len(j.preds["m"]) == 2 * len(truth)
    assert np.array_equal(j.rows["m"], np.repeat(np.arange(len(truth)), 2))


def test_dedupe(truth):
    p = truth.particles
    first = p[:, :1]
    dup = ak.with_field(truth, ak.concatenate([p, first], axis=1), "particles")
    out, n = io.dedupe_objects(dup)
    assert n == len(truth)
    assert ak.all(ak.num(out.particles) == ak.num(p))


@pytest.mark.real_data
@pytest.mark.skipif(not COCOA_VAL.exists(), reason="COCOA val file not available")
def test_real_cocoa_truth():
    truth = CocoaTruth(COCOA_VAL).read()
    assert len(truth) == 99_955
    assert set(truth.particles.fields) == {"pt", "eta", "phi", "e", "pdgid"}
    assert {"pt", "eta", "phi", "d0", "z0", "particle_idx"} == set(truth.tracks.fields)
    assert ak.sum(ak.num(truth.tracks)) > 0
    _, groups = io.duplicate_groups(truth)
    assert (groups * (groups - 1) // 2).sum() == 413  # 398 pairs + 5 triplets
    _, n_dup = io.dedupe_objects(truth)
    assert n_dup == 408
    j = io.join(truth, {"ident": synthetic.identical(truth)})
    assert len(j.event_ids) == 99_955
    rows = {(r["step"], r["model"]): r for r in j.cutflow}
    assert rows[("join", "ident")]["missing_events"] == 0
    # the event list of the task is a subset of the file
    ids = io.read_event_list("data/pflow_val_v1_events.txt")
    assert len(ids) == 20_000 and np.isin(ids, ak.to_numpy(truth.event_id)).all()
