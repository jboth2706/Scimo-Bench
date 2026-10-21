import awkward as ak
import numpy as np
import pandas as pd
import pytest

from scimo_bench import cutflow, io, scorer
from scimo_bench.datasets.cocoa import CATALOGUE as COCOA
from scimo_bench.scorer import (
    IQR,
    Efficiency,
    FakeRate,
    Mean,
    Median,
    ModelInfo,
    Quantile,
    Sample,
    Std,
    weighted_quantile,
)
from tests import synthetic

PARAMS = dict(match_dr=0.05, emd_r=1.0)
EDGES = [0, 40, 80, 120, 160, 200, 1000]


def score(truth, preds, metrics, n_boot=50, **kw):
    j = io.join(truth, preds)
    infos = [ModelInfo(m, "c") for m in preds]
    return j, scorer.score(j, infos, metrics, COCOA, PARAMS, n_boot=n_boot, **kw)


@pytest.mark.parametrize("q", [0.0, 0.1, 0.25, 0.5, 0.77, 0.9, 1.0])
def test_weighted_quantile_equals_numpy(q):
    rng = np.random.default_rng(0)
    v = rng.normal(size=37)
    s = Sample.build(v, np.arange(37))
    assert weighted_quantile(s, np.ones(37, dtype=int), q) == pytest.approx(np.quantile(v, q))
    w = rng.integers(0, 4, 37)
    expanded = np.repeat(s.values, w)
    assert weighted_quantile(s, w, q) == pytest.approx(np.quantile(expanded, q))


def test_values_match_numpy_references(truth):
    pred = synthetic.smeared(truth, 0.1)
    metrics = [
        Median("med", "jet_e_response"),
        IQR("iqr", "jet_e_response"),
        Mean("mean", "jet_e_response"),
        Std("std", "jet_e_response"),
        Quantile("q90", "jet_e_response", q=0.9),
        Median("bmed", "jet_e_response", x="truth.jet_pt", binning=EDGES),
        Efficiency("eff", "particle_matched", x="truth.particle_pt", binning=[1, 5, 10, 1000]),
        FakeRate("fake", "particle_fake"),
    ]
    j, res = score(truth, {"m": pred}, metrics)
    r = res.report.set_index(["metric", "bin_low"])
    e_pred = ak.sum(ak.values_astype(pred.particles.e, np.float64), axis=1)
    e_truth = ak.sum(ak.values_astype(truth.particles.e, np.float64), axis=1)
    resp = ak.to_numpy(e_pred / e_truth)
    ref = {
        "med": np.median(resp),
        "iqr": np.subtract(*np.quantile(resp, [0.75, 0.25])),
        "mean": resp.mean(),
        "std": resp.std(),
        "q90": np.quantile(resp, 0.9),
    }
    for name, v in ref.items():
        assert r.loc[(name, np.nan), "value"] == pytest.approx(v, rel=1e-9), name
    jpt = res.values["m"]["truth.jet_pt"]
    for lo, hi in zip(EDGES[:-1], EDGES[1:], strict=True):
        sel = (jpt >= lo) & (jpt < hi)
        row = r.loc[("bmed", float(lo))]
        if sel.any():
            assert row["value"] == pytest.approx(np.median(resp[sel]))
            assert row["status"] == "ok"
            assert row["lower"] <= row["value"] <= row["upper"]
        else:
            assert row["status"] == "invalid"
    # all matched within 0.05 (angles untouched) -> efficiency 1 in every bin, no fakes
    assert (res.report[res.report.metric == "eff"]["value"] == 1.0).all()
    assert r.loc[("fake", np.nan), "value"] == 0.0


def test_empty_bin_is_invalid(truth):
    m = Median("m", "jet_e_response", x="truth.jet_pt", binning=[1e5, 2e5])
    _, res = score(truth, {"a": synthetic.identical(truth)}, [m])
    binned = res.report[res.report.bin_low.notna()]
    assert (binned.status == "invalid").all() and binned.value.isna().all()


def test_missing_field_does_not_abort(truth):
    preds = {
        "ok": synthetic.identical(truth),
        "noe": synthetic.missing_field(synthetic.identical(truth), "e"),
    }
    metrics = [
        Median("resp", "jet_e_response", x="truth.jet_pt", binning=EDGES),
        Median("emd", "event_emd"),
    ]
    j = io.join(truth, preds)
    infos = [ModelInfo("ok", "c"), ModelInfo("noe", "c"), ModelInfo("gone", "c", available=False)]
    res = scorer.score(j, infos, metrics, COCOA, PARAMS, n_boot=20)
    rep = res.report
    noe = rep[rep.model == "noe"]
    assert (noe[noe.metric == "resp"].status == "missing").all()
    assert (noe[noe.metric == "emd"].status == "ok").all()
    assert (rep[rep.model == "gone"].status == "missing").all()
    assert len(rep[rep.model == "gone"]) == len(rep[rep.model == "ok"])
    assert list(rep.columns) == scorer.REPORT_COLUMNS
    assert set(rep.kind) == {"quality"}


def test_paired_bootstrap_and_reproducibility(truth):
    preds = {"a": synthetic.smeared(truth, 0.1, seed=1), "b": synthetic.smeared(truth, 0.2, seed=2)}
    metrics = [Median("med", "jet_e_response"), IQR("iqr", "jet_e_response")]
    _, r1 = score(truth, preds, metrics, n_boot=100, seed=7)
    _, r2 = score(truth, preds, metrics, n_boot=100, seed=7)
    pd.testing.assert_frame_equal(r1.report, r2.report)
    _, r3 = score(truth, preds, metrics, n_boot=100, seed=8)
    assert not np.allclose(r1.report.lower, r3.report.lower)
    # a model scored alone gets the same interval as when paired (same resamples)
    _, alone = score(truth, {"a": preds["a"]}, metrics, n_boot=100, seed=7)
    pd.testing.assert_frame_equal(
        alone.report.reset_index(drop=True),
        r1.report[r1.report.model == "a"].reset_index(drop=True),
    )


def test_bootstrap_coverage():
    """95 % intervals of the median cover the true median in ~95 % of 200 toys."""
    rng = np.random.default_rng(123)
    n_events, n_toys, n_boot = 300, 200, 400
    true_median = 1.0  # lognormal(0, 0.5) has median exp(0) = 1
    med = Median("m", "v")
    covered = 0
    for _ in range(n_toys):
        v = rng.lognormal(0.0, 0.5, n_events)
        s = Sample.build(v, np.arange(n_events))
        boots = [
            med(s, np.bincount(rng.integers(0, n_events, n_events), minlength=n_events))
            for _ in range(n_boot)
        ]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        covered += lo <= true_median <= hi
    assert covered / n_toys == pytest.approx(0.95, abs=0.03)


def test_rankings(truth):
    preds = {
        "good": synthetic.smeared(truth, 0.05, seed=1),
        "bad": synthetic.smeared(truth, 0.3, seed=2),
        "noe": synthetic.missing_field(synthetic.smeared(truth, 0.01, seed=3), "e"),
    }
    metrics = [
        IQR("iqr", "jet_e_response"),
        Median("med", "jet_e_response", better={"closest_to": 1.0}),
        Mean("mult", "reco.n_particles", better="higher"),
    ]
    _, res = score(truth, preds, metrics, n_boot=10)
    rk = scorer.rank(res.report, metrics, primary="iqr")
    r = rk.set_index(["metric", "model"])
    assert r.loc[("iqr", "good"), "rank"] == 1 and r.loc[("iqr", "bad"), "rank"] == 2
    assert r.loc[("iqr", "noe"), "rank"] == 3 and r.loc[("iqr", "noe"), "status"] == "missing"
    assert r.loc[("iqr", "good"), "primary"] and not r.loc[("med", "good"), "primary"]
    # equal multiplicity -> tie at rank 1 for all three
    assert {r.loc[("mult", m), "rank"] for m in preds} == {1}
    assert r.loc[("mean_rank", "good"), "rank"] == 1
    assert r.loc[("mean_rank", "noe"), "status"] == "missing"
    assert list(rk.columns) == scorer.RANKING_COLUMNS


# 30 % smearing gives a few negative pT values; the EMD (computed in every run) warns on them
@pytest.mark.filterwarnings("ignore:Problem infeasible")
def test_normalized_score(truth):
    preds = {
        "base": synthetic.smeared(truth, 0.3, seed=1),
        "ref": synthetic.identical(truth),
        "mid": synthetic.smeared(truth, 0.15, seed=2),
    }
    metrics = [IQR("iqr", "jet_e_response")]
    _, res = score(truth, preds, metrics, n_boot=5)
    rk = scorer.rank(res.report, metrics, normalization={"baseline": "base", "reference": "ref"})
    n = rk[rk.metric == "iqr"].set_index("model")["normalized"]
    assert n["base"] == pytest.approx(0.0) and n["ref"] == pytest.approx(1.0)
    assert 0 < n["mid"] < 1
    rk = scorer.rank(res.report, metrics, normalization={"baseline": "x", "reference": "ref"})
    assert rk["normalized"].isna().all()


def test_better_validation():
    with pytest.raises(ValueError, match="better"):
        Median("m", "v", better="smaller")
    with pytest.raises(ValueError, match="binning"):
        Median("m", "v", x="w")
    with pytest.raises(ValueError, match="increasing"):
        Median("m", "v", x="w", binning=[2, 1])


def test_per_event_table(truth):
    preds = {
        "a": synthetic.identical(truth),
        "noe": synthetic.missing_field(synthetic.identical(truth)),
    }
    j = io.join(truth, preds)
    j = cutflow.apply(j, [cutflow.CutStep("min_particles", 3, "truth")])
    res = scorer.score(
        j,
        [ModelInfo("a", "c"), ModelInfo("noe", "c")],
        [Median("r", "jet_e_response")],
        COCOA,
        PARAMS,
        n_boot=0,
    )
    pe = res.per_event
    assert set(pe.fields) >= {
        "model",
        "event_id",
        "sample_id",
        "jet_e_response",
        "particle_pt",
        "jet_e",
    }
    assert ak.sum(pe.model == "a") == len(j.event_ids) == ak.sum(pe.model == "truth")
    noe = pe[pe.model == "noe"]
    assert ak.all(ak.is_none(noe.jet_e_response))
    assert ak.all(ak.num(pe[pe.model == "a"].particle_pt) >= 3)
    # every observable on both sides; comparisons for models only
    assert ak.all(ak.is_none(pe[pe.model == "truth"].jet_e_response))
    assert not ak.any(ak.is_none(pe[pe.model == "truth"].jet_e))
    assert ak.all(ak.is_none(noe.jet_e))
