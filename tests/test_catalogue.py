import ast
import types
from pathlib import Path

import awkward as ak
import numpy as np
import pytest

from scimo_bench import catalogue, io
from scimo_bench.catalogue import Catalogue, Missing, comparison, observable, plot
from scimo_bench.datasets.cocoa import CATALOGUE as COCOA
from scimo_bench.scorer import evaluate
from tests import synthetic

PARAMS = {}  # the pflow task's values (its comparisons need none)


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
    resp, ptr, dr, n_diff = compare(
        truth,
        synthetic.identical(truth),
        "jet_e_response",
        "jet_pt_response",
        "jet_dr",
        "multiplicity_diff",
    )
    assert resp == pytest.approx(1.0)
    assert ptr == pytest.approx(1.0)
    assert np.allclose(dr, 0)
    assert (n_diff == 0).all()


def test_shifted_pt(truth):
    resp, ptr, dr = compare(
        truth, synthetic.shifted(truth, 1.1), "jet_e_response", "jet_pt_response", "jet_dr"
    )
    assert resp == pytest.approx(1.1, rel=1e-5)
    assert ptr == pytest.approx(1.1, rel=1e-5)
    assert np.allclose(dr, 0, atol=1e-6)  # a common pT scale keeps the jet axis


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
    assert COCOA.params() == set()
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
