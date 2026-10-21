import awkward as ak
import numpy as np
import pandas as pd
import pytest

from scimo_bench import cutflow, io
from scimo_bench.cutflow import CutStep
from tests import synthetic

CENTRAL = [
    CutStep("max_abs_eta", 2.5, "both"),
    CutStep("min_pt", 1.0, "both"),
    CutStep("min_particles", 1, "truth"),
]


def test_object_and_event_cuts(truth):
    pred_a = synthetic.smeared(truth, 0.3)
    pred_b = synthetic.dropped(truth, 0.5)
    j = io.join(truth, {"a": pred_a, "b": pred_b})
    out = cutflow.apply(j, [CutStep("min_pt", 3.0, "both"), CutStep("min_particles", 2, "truth")])
    t_pt = out.truth.particles.pt
    assert ak.all(t_pt > 3.0)
    assert ak.all(ak.num(t_pt) >= 2)
    for m in ("a", "b"):
        assert ak.all(out.preds[m].particles.pt > 3.0)
        assert np.array_equal(out.preds[m].event_id, out.event_ids)  # same events for all
    # expected counts by hand
    pts = truth.particles.pt
    n_after = ak.to_numpy(ak.sum(pts > 3.0, axis=1))
    assert len(out.event_ids) == (n_after >= 2).sum()
    df = cutflow.to_frame(out.cutflow)
    ev = df[df.step == "min_particles"].set_index("model")["events"]
    assert ev["truth"] == ev["a"] == ev["b"] == (n_after >= 2).sum()


def test_preselection_counts_equal_for_models(truth):
    j = io.join(truth, {"x": synthetic.identical(truth), "y": synthetic.shifted(truth, 0.5)})
    out = cutflow.apply(j, CENTRAL)
    df = cutflow.to_frame(out.cutflow)
    last = df[df.step == "min_particles"].set_index("model")
    assert last.loc["x", "events"] == last.loc["y", "events"] == last.loc["truth", "events"]
    # identical model keeps exactly the truth objects; the shifted one loses soft ones
    assert last.loc["x", "objects"] == last.loc["truth", "objects"]
    assert last.loc["y", "objects"] < last.loc["truth", "objects"]
    assert list(df.columns) == cutflow.CUTFLOW_COLUMNS
    assert isinstance(df, pd.DataFrame)


def test_event_cut_reindexes_multi_sample_rows(truth):
    pred = io.normalize(
        ak.concatenate(
            [
                synthetic.identical(truth),
                ak.with_field(
                    synthetic.identical(truth), np.ones(len(truth), dtype=np.int64), "sample_id"
                ),
            ]
        )
    )
    j = io.join(truth, {"m": pred})
    out = cutflow.apply(j, [CutStep("min_particles", 5, "truth")])
    assert len(out.preds["m"]) == 2 * len(out.event_ids)
    assert np.array_equal(out.event_ids[out.rows["m"]], ak.to_numpy(out.preds["m"].event_id))


def test_unknown_cut_is_config_error():
    with pytest.raises(ValueError, match="unknown cut"):
        cutflow.check([CutStep("max_pt", 1.0, "both")])
    with pytest.raises(ValueError, match="on must be one of"):
        cutflow.check([CutStep("min_pt", 1.0, "model")])
    # removing events for one model only would break the paired comparison
    with pytest.raises(ValueError, match="must be on: truth"):
        cutflow.check([CutStep("min_particles", 1, "reco")])


def test_cut_only_one_side(truth):
    pred = synthetic.identical(truth)
    j = io.join(truth, {"m": pred})
    n_all = ak.sum(ak.num(truth.particles))
    on_truth = cutflow.apply(j, [CutStep("min_pt", 3.0, "truth")])
    assert ak.all(on_truth.truth.particles.pt > 3.0)
    assert ak.sum(ak.num(on_truth.preds["m"].particles)) == n_all  # predictions untouched
    on_reco = cutflow.apply(j, [CutStep("min_pt", 3.0, "reco")])
    assert ak.all(on_reco.preds["m"].particles.pt > 3.0)
    assert ak.sum(ak.num(on_reco.truth.particles)) == n_all  # truth untouched
    df = cutflow.to_frame(on_reco.cutflow)
    row = df[df.step == "min_pt"].set_index("model")
    assert set(row["on"]) == {"reco"} and row.loc["truth", "dropped"] == 0
    assert row.loc["m", "dropped"] > 0
