import ast
import types
from pathlib import Path

import awkward as ak
import numpy as np
import pytest

from scimo_bench import catalogue, io
from scimo_bench.catalogue import Catalogue, Missing, comparison, observable, plot
from scimo_bench.datasets.cocoa import CATALOGUE as COCOA
from scimo_bench.datasets.common import emd_event
from scimo_bench.scorer import evaluate
from tests import synthetic

PARAMS = dict(match_dr=0.05, emd_r=1.0)  # the pflow task's values


def compare(truth, reco, *names, **params):
    """Comparisons of one model against the truth, on joined rows."""
    j = io.join(truth, {"m": reco})
    t, r = j.truth_rows("m"), j.preds["m"]
    return [
        catalogue.call_comparison(COCOA.comparisons[n], t, r, {**PARAMS, **params}) for n in names
    ]


def flat(x):
    return ak.to_numpy(ak.flatten(x)) if isinstance(x, ak.Array) else np.asarray(x)


def test_identical(truth):
    resp, ptr, emd, matched, fake, dr, n_diff, pr = compare(
        truth,
        synthetic.identical(truth),
        "jet_e_response",
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
    resp, ptr, pr, emd = compare(
        truth,
        synthetic.shifted(truth, 1.1),
        "jet_e_response",
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
    matched, fake = compare(truth, synthetic.dropped(truth, p), "particle_matched", "particle_fake")
    eff = flat(matched).mean()
    n = len(flat(matched))
    assert eff == pytest.approx(1 - p, abs=3 * np.sqrt(p * (1 - p) / n))
    assert not flat(fake).any()


def test_observables_run_on_both_sides(truth):
    """One observable function gives the truth value and each model's reco value."""
    j = io.join(truth, {"m": synthetic.shifted(truth, 1.1)})
    vals = evaluate(j, COCOA, PARAMS)
    t, r = vals["m"]["truth.jet_e"], vals["m"]["reco.jet_e"]
    assert r == pytest.approx(1.1 * t, rel=1e-5)
    assert vals["truth"]["jet_e"] == pytest.approx(t)


def test_missing_field_is_missing_and_warns(truth, caplog):
    """A field models must output (particles.e) → Missing with a warning; quantities that
    do not need it are still computed."""
    reco = synthetic.missing_field(synthetic.identical(truth), "e")
    j = io.join(truth, {"m": reco})
    vals = evaluate(j, COCOA, PARAMS, required={"particles", "pt", "eta", "phi", "e"})
    assert isinstance(vals["m"]["reco.jet_e"], Missing)
    assert isinstance(vals["m"]["jet_e_response"], Missing)
    assert vals["m"]["jet_pt_response"] == pytest.approx(1.0)
    assert vals["m"]["truth.jet_e"].min() > 0
    assert "jet_e_response is missing" in caplog.text


def test_input_collection_is_truth_only(truth, caplog):
    """Tracks are in the truth only (a model input): computed for the truth, silently
    missing for the models."""
    tracks = ak.zip({"pt": truth.particles.pt, "eta": truth.particles.eta})
    with_tracks = ak.with_field(truth, tracks, "tracks")
    j = io.join(with_tracks, {"m": synthetic.identical(truth)})
    vals = evaluate(j, COCOA, PARAMS, required={"particles", "pt", "eta", "phi", "e"})
    assert np.array_equal(vals["truth"]["n_tracks"], ak.to_numpy(ak.num(truth.particles)))
    assert isinstance(vals["m"]["reco.n_tracks"], Missing)
    assert "n_tracks" not in caplog.text


def one_event(t, r):
    """Arrays for one event from lists of (pt, eta, phi)."""

    def arr(objs):
        objs = np.array(objs, dtype=np.float64).reshape(-1, 3)
        return ak.zip(
            {
                "event_id": np.array([1]),
                "sample_id": np.array([0]),
                "particles": ak.zip(
                    {k: ak.Array([objs[:, i].tolist()]) for i, k in enumerate(("pt", "eta", "phi"))}
                ),
            },
            depth_limit=1,
        )

    return arr(t), arr(r)


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
    # nothing reconstructed: all pT goes to ground at cost R
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
    # two truth particles compete for one reco particle: the harder one wins
    t, r = one_event([(5, 0.0, 0.0), (20, 0.03, 0.0)], [(10, 0.02, 0.0)])
    matched, dr = compare(t, r, "particle_matched", "particle_dr")
    assert flat(matched).tolist() == [0.0, 1.0]
    assert flat(dr)[1] == pytest.approx(0.01)
    # outside the threshold: unmatched truth, fake reco
    t, r = one_event([(5, 0.0, 0.0)], [(5, 0.2, 0.0)])
    matched, fake = compare(t, r, "particle_matched", "particle_fake")
    assert flat(matched).tolist() == [0.0] and flat(fake).tolist() == [1.0]
    # a task parameter overrides the default
    (matched,) = compare(t, r, "particle_matched", match_dr=0.3)
    assert flat(matched).tolist() == [1.0]


def test_jet_massless_vector_sum():
    t, _ = one_event([(10, 0.0, 0.0), (10, 0.0, np.pi / 2)], [])
    obs = COCOA.observables
    assert obs["jet_pt"](t)[0] == pytest.approx(10 * np.sqrt(2))
    assert obs["jet_phi"](t)[0] == pytest.approx(np.pi / 4)
    assert obs["jet_eta"](t)[0] == pytest.approx(0.0)


def test_catalogue_names_and_labels():
    assert COCOA.info("jet_e").axis_label() == "jet $E$ [GeV]"
    assert COCOA.info("truth.jet_e").axis_label("truth") == "jet $E$ (truth) [GeV]"
    assert COCOA.info("jet_e_response").reference == 1.0
    assert {"match_dr", "emd_r"} == COCOA.params()
    COCOA.check_name("truth.jet_pt", "x")
    COCOA.check_name("jet_e_response", "m")
    with pytest.raises(ValueError, match="truth.jet_pt or reco.jet_pt"):
        COCOA.check_name("jet_pt", "metric m")
    with pytest.raises(ValueError, match="unknown 'reco.nope'"):
        COCOA.check_name("reco.nope", "metric m")
    with pytest.raises(KeyError, match="unknown"):
        COCOA.info("nope")


def test_collect_only_own_decorated_functions():
    obs, comp, plots = (types.ModuleType(n) for n in ("o", "c", "p"))

    @observable(label="a")
    def a(ev):
        return ev

    @comparison()
    def b(truth, reco, k=1):
        return k

    @plot
    def c(p):
        return None

    for mod, fn in ((obs, a), (comp, b), (plots, c)):
        fn.__module__ = mod.__name__
        setattr(mod, fn.__name__, fn)
    comp.a = a  # imported from observables: not collected again
    comp.np = np  # np.info exists but is not a catalogue entry
    cat = Catalogue.collect(obs, comp, plots)
    assert list(cat.observables) == ["a"] and list(cat.comparisons) == ["b"]
    assert list(cat.plots) == ["c"] and cat.params() == {"k"}
    obs.b = types.FunctionType(b.__code__, {}, "b")
    obs.b.info, obs.b.__module__ = catalogue.Info("b", "observable"), "o"
    with pytest.raises(ValueError, match="observable and comparison"):
        Catalogue.collect(obs, comp, plots)


def test_no_physics_in_scorer_and_plotter():
    """Scorer and plotter import no physics helpers."""
    root = Path(catalogue.__file__).parent
    for module in ("scorer.py", "plotter.py"):
        tree = ast.parse((root / module).read_text())
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert not any(m.startswith("scimo_bench.datasets") for m in imported), module
