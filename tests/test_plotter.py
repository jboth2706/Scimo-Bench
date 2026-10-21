import awkward as ak
import numpy as np
import pandas as pd
import pytest

from scimo_bench import plotter
from scimo_bench.datasets.cocoa import CATALOGUE as COCOA
from scimo_bench.plotter import model_styles


def test_styles_are_stable_and_distinct():
    names = ["nanohep_n6_adamw", "nanohep_n6_muon", "nanohep_depth_n5_adamw", "dummy"]
    a = model_styles(names)
    assert a == model_styles(list(reversed(names)))
    assert len({s["color"] for s in a.values()}) == len(names)
    assert all(s["color"] in plotter.Style().palette for s in a.values())
    many = model_styles([f"m{i}" for i in range(8)])
    assert len({s["color"] for s in many.values()}) == 8
    more = model_styles([f"m{i}" for i in range(10)])
    assert len({(s["color"], s["linestyle"]) for s in more.values()}) == 10


def context(tmp_path, per_event=None, models=("a",)):
    return plotter.PlotContext(
        folder=tmp_path,
        report=pd.DataFrame(columns=["model", "metric", "bin_low", "bin_high", "status", "kind"]),
        rankings=None,
        per_event=per_event,
        models=list(models),
        labels={},
        metrics={},
        catalogue=COCOA,
        formats=["png"],
    )


def test_plots_without_data_are_skipped(tmp_path):
    pc = context(tmp_path)
    plots = [COCOA.plots["jet_e"], COCOA.plots["response_vs_pt"], plotter.ranking]
    assert plotter.draw(pc, plots) == [] and not (tmp_path / "pareto.csv").exists()


def per_event():
    """Two events: truth and model `a` with jet_e, truth only with n_tracks."""
    return ak.Array(
        [
            dict(model="truth", jet_e=50.0, n_tracks=3.0, jet_e_response=None),
            dict(model="truth", jet_e=60.0, n_tracks=4.0, jet_e_response=None),
            dict(model="a", jet_e=55.0, n_tracks=None, jet_e_response=1.1),
            dict(model="a", jet_e=66.0, n_tracks=None, jet_e_response=1.1),
        ]
    )


def labels(pc, name, bins):
    fig, ax = pc.figure()
    pc.hist(ax, name, bins)
    return ax.get_xlabel(), ax.get_legend_handles_labels()[1]


def test_hist_labels_say_truth_or_reco(tmp_path):
    pc = context(tmp_path, per_event())
    bins = np.linspace(0, 100, 11)
    # both levels: plain axis, the legend names them
    assert labels(pc, "jet_e", bins) == ("jet $E$ [GeV]", ["truth", "a (reco)"])
    # truth only: the axis says so
    assert labels(pc, "n_tracks", bins) == ("tracks per event (truth)", ["truth"])
    # models only (truth=False): the axis says reco
    fig, ax = pc.figure()
    pc.hist(ax, "jet_e", bins, truth=False)
    assert ax.get_xlabel() == "jet $E$ (reco) [GeV]"
    # a comparison: its label names both levels, one line per model
    x, legend = labels(pc, "jet_e_response", np.linspace(0, 2, 11))
    assert "reco" in x and "truth" in x and legend == ["a"]


def test_plot_functions_are_saved_under_their_name(tmp_path):
    pc = context(tmp_path, per_event())
    paths = plotter.draw(pc, [COCOA.plots["jet_e"], COCOA.plots["n_tracks"]])
    assert sorted(p.name for p in paths) == ["jet_e.png", "n_tracks.png"]


def test_model_style_overrides_and_style_keys():
    st = model_styles(["a", "b"], overrides={"a": {"color": "navy", "linestyle": None}})
    assert st["a"]["color"] == "navy" and st["a"]["linestyle"] in plotter.Style().linestyles
    with pytest.raises(ValueError, match="colour 'redd'"):
        plotter.check_colour("redd", "models.a.color")
    with pytest.raises(ValueError, match="plots.style.truth_color"):
        plotter.Style.from_cfg({"truth_color": "greyish"})
    small = plotter.Style.from_cfg({"palette": ["#111111", "#222222"], "font_size": 12})
    assert set(s["color"] for s in model_styles(["a", "b"], small).values()) == {
        "#111111",
        "#222222",
    }
    assert small.rc_params()["font.size"] == 12
    with pytest.raises(ValueError, match="fontsize"):
        plotter.Style.from_cfg({"fontsize": 12})
