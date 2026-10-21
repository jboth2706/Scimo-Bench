import pandas as pd
import pytest

from scimo_bench import plotter
from scimo_bench.plotter import Distribution, MetricVsX, RankingPlot, model_styles, resolve


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


def test_resolve_plot_names():
    spec = Distribution("jet_pred_e", bins=[0, 1, 2])
    assert resolve("jet_pred_e", {"jet_pred_e": spec}) is spec
    assert isinstance(resolve("ranking", {}), RankingPlot)
    d = resolve("event_emd", {})
    assert isinstance(d, Distribution) and d.variable == "event_emd"
    assert resolve("not_a_thing", {}) is None


def test_plots_without_data_are_skipped(tmp_path):
    pc = plotter.PlotContext(
        folder=tmp_path,
        report=pd.DataFrame(columns=["model", "metric", "bin_low", "bin_high", "status", "kind"]),
        rankings=None,
        per_event=None,
        models=["a"],
        labels={},
        metrics={},
    )
    paths = plotter.draw(pc, [Distribution("event_emd"), MetricVsX("x"), RankingPlot()])
    assert paths == [] and not (tmp_path / "pareto.csv").exists()


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
