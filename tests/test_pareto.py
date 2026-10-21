import numpy as np
import pandas as pd

from scimo_bench.plotter import ParetoPlot, pareto_front, parse_condition
from scimo_bench.scorer import REPORT_COLUMNS


def test_known_front_with_dominated_point_and_ties():
    #          A    B    C (dominated by B)  D (tie with A)  E (worse y, cheapest)
    x = np.array([1.0, 2.0, 3.0, 1.0, 0.5])
    y = np.array([5.0, 3.0, 4.0, 5.0, 9.0])
    assert pareto_front(x, y, "lower").tolist() == [True, True, False, True, True]


def test_front_with_higher_and_closest_to():
    x = np.array([1.0, 2.0, 3.0])
    assert pareto_front(x, [0.5, 0.9, 0.8], "higher").tolist() == [True, True, False]
    assert pareto_front(x, [1.3, 0.95, 1.01], {"closest_to": 1.0}).tolist() == [True, True, True]
    assert pareto_front(x, [1.0, 1.2, 0.9], {"closest_to": 1.0}).tolist() == [True, False, False]


def test_robust_dominance_needs_separated_intervals():
    x, y = np.array([1.0, 2.0]), np.array([1.0, 1.1])
    assert pareto_front(x, y).tolist() == [True, False]
    wide = pareto_front(x, y, "lower", (x - 0.1, x + 0.1), (y - 0.5, y + 0.5), robust=True)
    assert wide.tolist() == [True, True]
    tight = pareto_front(x, y, "lower", (x - 0.1, x + 0.1), (y - 0.01, y + 0.01), robust=True)
    assert tight.tolist() == [True, False]


def row(model, metric, value, kind, condition, status="ok"):
    return dict(
        model=model,
        metric=metric,
        bin_low=np.nan,
        bin_high=np.nan,
        value=value,
        lower=value * 0.9,
        upper=value * 1.1,
        status=status,
        kind=kind,
        task_version=1,
        condition=condition,
    )


def test_join_quality_and_perf_rows():
    perf_c = (
        "gen={g};scenario=single;batch_size=1;device=NVIDIA A100-SXM4-40GB;"
        "precision=bf16;compiled=True"
    )
    rows = [
        row("a", "emd_median", 2.0, "quality", "gen=g1"),
        row("a", "emd_median", 1.5, "quality", "gen=g2"),
        row("b", "emd_median", 3.0, "quality", "gen=g3"),
        row("a", "latency_p50", 10.0, "perf", perf_c.format(g="g1")),
        row("a", "latency_p50", 30.0, "perf", perf_c.format(g="g2")),
        row("b", "latency_p50", 5.0, "perf", perf_c.format(g="g3")),
        # other batch size, other device, invalid gate: all excluded
        row(
            "b",
            "latency_p50",
            1.0,
            "perf",
            perf_c.format(g="g3").replace("batch_size=1", "batch_size=64"),
        ),
        row("b", "latency_p50", 1.0, "perf", perf_c.format(g="g3").replace("A100", "H100")),
        row("a", "latency_p50", 1.0, "perf", perf_c.format(g="g1") + ";x=1", status="invalid"),
    ]
    rep = pd.DataFrame(rows, columns=REPORT_COLUMNS)
    p = ParetoPlot(
        "emd_median", "latency_p50", dict(scenario="single", batch_size=1, device="A100")
    )
    pts = p.points(rep)
    assert sorted(zip(pts.model, pts.x, pts.y, strict=True)) == [
        ("a", 10.0, 2.0),
        ("a", 30.0, 1.5),
        ("b", 5.0, 3.0),
    ]
    assert parse_condition(perf_c.format(g="g1"))["device"] == "NVIDIA A100-SXM4-40GB"


def test_no_perf_rows_gives_no_points():
    rep = pd.DataFrame([row("a", "emd_median", 2.0, "quality", "gen=g1")], columns=REPORT_COLUMNS)
    pts = ParetoPlot("emd_median", "latency_p50", dict(scenario="single")).points(rep)
    assert pts.empty
