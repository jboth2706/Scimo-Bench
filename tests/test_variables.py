import ast
from pathlib import Path

import awkward as ak
import numpy as np
import pytest

from scimo_bench import io, variables
from scimo_bench.variables import Context, MissingInput, emd_event, get
from tests import synthetic

PARAMS = dict(match_dr=0.05, emd_r=1.0)  # the pflow task's values


def make_ctx(**params):
    return Context(params={**PARAMS, **params})


def evaluate(truth, pred, *names, **params):
    j = io.join(truth, {"m": pred})
    c = make_ctx(**params)
    t, p = j.truth_rows("m"), j.preds["m"]
    return [get(n)(t, p, ctx=c) for n in names]


def flat(x):
    return ak.to_numpy(ak.flatten(x)) if isinstance(x, ak.Array) else np.asarray(x)


def test_identical(truth):
    resp, ptr, emd, matched, fake, dr, n_diff, pr = evaluate(
        truth,
        synthetic.identical(truth),
        "jet_energy_response",
        "jet_pt_response",
        "event_emd",
        "particle_matched",
        "particle_fake",
        "jet_dr",
        "multiplicity_diff",
        "particle_pt_response",
    )
    assert resp == pytest.approx(1.0)
    assert ptr == pytest.approx(1.0)
    assert np.abs(emd).max() < 1e-6
    assert flat(matched).all() and not flat(fake).any()
    assert np.allclose(dr, 0)
    assert (n_diff == 0).all()
    assert flat(pr) == pytest.approx(1.0)


def test_shifted_pt(truth):
    resp, ptr, pr, emd = evaluate(
        truth,
        synthetic.shifted(truth, 1.1),
        "jet_energy_response",
        "jet_pt_response",
        "particle_pt_response",
        "event_emd",
    )
    assert resp == pytest.approx(1.1, rel=1e-5)
    assert ptr == pytest.approx(1.1, rel=1e-5)
    assert flat(pr) == pytest.approx(1.1, rel=1e-5)
    # EMD of a pure pT excess: 10 % of the truth pT moved to the ground particle at cost R=1
    t_sum = ak.to_numpy(ak.sum(truth.particles.pt, axis=1))
    assert emd == pytest.approx(0.1 * t_sum, rel=1e-4)


def test_dropped_particles_efficiency(truth):
    p = 0.2
    matched, fake = evaluate(
        truth, synthetic.dropped(truth, p), "particle_matched", "particle_fake"
    )
    eff = flat(matched).mean()
    n = len(flat(matched))
    assert eff == pytest.approx(1 - p, abs=3 * np.sqrt(p * (1 - p) / n))
    assert not flat(fake).any()


def test_missing_energy_field(truth):
    pred = synthetic.missing_field(synthetic.identical(truth), "e")
    j = io.join(truth, {"m": pred})
    c = make_ctx()
    t, pr = j.truth_rows("m"), j.preds["m"]
    for name in ("jet_pred_e", "jet_energy_response"):
        with pytest.raises(MissingInput, match="particles.e"):
            get(name)(t, pr, ctx=c)
    assert get("jet_pt_response")(t, pr, ctx=c) == pytest.approx(1.0)
    assert get("jet_truth_e")(t, pr, ctx=c).min() > 0


def test_missing_collection(truth):
    pred = synthetic.identical(truth)[["event_id", "sample_id"]]
    with pytest.raises(MissingInput, match="particles"):
        get("n_pred")(truth, pred, ctx=make_ctx())


def one_event(t, p):
    """Arrays for one event from lists of (pt, eta, phi)."""

    def arr(objs):
        objs = np.array(objs, dtype=np.float64).reshape(-1, 3)
        return ak.zip(
            {
                "event_id": np.array([1]),
                "particles": ak.zip(
                    {k: ak.Array([objs[:, i].tolist()]) for i, k in enumerate(("pt", "eta", "phi"))}
                ),
            },
            depth_limit=1,
        )

    return arr(t), arr(p)


def test_emd_known_values():
    # one particle moved by ΔR = 0.1 carries all its pT that distance
    assert emd_event(
        np.array([10.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([10.0]),
        np.array([0.1]),
        np.array([0.0]),
    ) == pytest.approx(1.0)
    # beyond R the cost is capped
    assert emd_event(
        np.array([10.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([10.0]),
        np.array([2.0]),
        np.array([0.0]),
    ) == pytest.approx(10.0)
    # nothing predicted: all pT goes to ground at cost R
    assert emd_event(
        np.array([3.0, 4.0]), np.zeros(2), np.zeros(2), np.zeros(0), np.zeros(0), np.zeros(0)
    ) == pytest.approx(7.0)
    assert emd_event(*(np.zeros(0),) * 6) == 0.0
    # φ wraps around ±π
    assert emd_event(
        np.array([5.0]),
        np.array([0.0]),
        np.array([3.1]),
        np.array([5.0]),
        np.array([0.0]),
        np.array([-3.1]),
    ) == pytest.approx(5.0 * (2 * np.pi - 6.2), rel=1e-6)


def test_matching_is_greedy_by_truth_pt():
    # two truth particles compete for one prediction: the harder one wins
    t, p = one_event([(5, 0.0, 0.0), (20, 0.03, 0.0)], [(10, 0.02, 0.0)])
    c = make_ctx()
    matched = flat(get("particle_matched")(t, p, ctx=c))
    assert matched.tolist() == [0.0, 1.0]
    assert flat(get("particle_dr")(t, p, ctx=c))[1] == pytest.approx(0.01)
    # outside the threshold: unmatched truth, fake prediction
    t, p = one_event([(5, 0.0, 0.0)], [(5, 0.2, 0.0)])
    c = make_ctx()
    assert flat(get("particle_matched")(t, p, ctx=c)).tolist() == [0.0]
    assert flat(get("particle_fake")(t, p, ctx=c)).tolist() == [1.0]


def test_jet_massless_vector_sum():
    t, _ = one_event([(10, 0.0, 0.0), (10, 0.0, np.pi / 2)], [])
    c = make_ctx()
    assert get("jet_truth_pt")(t, None, ctx=c)[0] == pytest.approx(10 * np.sqrt(2))
    assert get("jet_truth_phi")(t, None, ctx=c)[0] == pytest.approx(np.pi / 4)
    assert get("jet_truth_eta")(t, None, ctx=c)[0] == pytest.approx(0.0)


def test_registry_metadata():
    info = variables.info("jet_pred_e")
    assert info.truth == "jet_truth_e" and info.unit == "GeV" and "[GeV]" in info.axis_label
    assert variables.info("n_truth").truth_only
    with pytest.raises(KeyError, match="unknown variable"):
        get("nope")
    with pytest.raises(ValueError, match="twice"):
        variables.GENERIC.variable("n_pred")(lambda t, p, c: None)
    with pytest.raises(KeyError, match="match_dr"):
        Context().param("match_dr")


def test_no_physics_outside_variables():
    """Scorer and plotter call variables, they do not import numpy physics helpers from it."""
    root = Path(variables.__file__).parent
    tree = ast.parse((root / "plotter.py").read_text())
    names = {
        a.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "scimo_bench.variables"
        for a in node.names
    }
    assert names <= {"info", "exists"}
