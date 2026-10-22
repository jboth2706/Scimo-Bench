"""Plot configs (group `plots`): which plots are drawn and how they look.

- `plots.draw`: names drawn (default: the task's `plots`); `plots.extra` adds names,
  `plots.skip` removes some. A name is a plot function of the dataset (its `plots.py`) or a
  generic plot (`ranking`). Binning, scales and labels are set in the plot function.
- `plots.style`: look of all plots (colours, line styles, sizes, fonts, matplotlib rc).
- `plots.pareto` (subgroup `plots/pareto`): a quality-vs-cost plot; `plots=pareto` adds one.

Per-model colour / line style / label: `models.<name>.color`, `.linestyle`, `.label`.
"""

from hydra_zen import builds, make_config, store

from scimo_bench.plotter import ParetoPlot

plots_store = store(group="plots")
pareto_store = store(group="plots/pareto")

STYLE = dict(
    # categorical order validated for colour-vision deficiency; fixed, never cycled
    palette=[
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#6250d6",
        "#e34948",
    ],
    linestyles=["-", "--", "-.", ":"],
    truth_color="#b4b3ad",
    ink="#0b0b0b",
    muted="#52514e",
    grid_color="grey",
    grid_linestyle=":",
    grid_alpha=0.3,
    font_size=10,
    line_width=2.5,
    hist_line_width=1.5,  # model lines in histograms
    ratio_marker_size=3.5,  # points in ratio panels
    figsize=[6.0, 4.0],
    ratio_figsize=[6.0, 5.0],
    dpi=150,
    legend="above",  # above the axes, or any matplotlib `loc` (e.g. "best", "upper right")
    rc={},  # any matplotlib rcParams, applied last
)


def pareto(**kw):
    """A Pareto spec with every option visible."""
    return builds(ParetoPlot, **kw, populate_full_signature=True)


# the y axis of the jet-response-IQR Pareto plots
IQR_AXIS = dict(ylabel="IQR of jet energy response", ylim=[0.01, 0.2])

pareto_store(
    pareto(
        y="response_iqr",
        x="latency_p50",
        perf_condition=dict(scenario="single", batch_size=1),
        # one event of COCOA single jet = one jet; batch size 1 → ms per jet
        xlabel="median latency [ms / jet]",
        **IQR_AXIS,
        name="pareto_iqr_vs_latency",
    ),
    name="iqr_vs_latency",
)
pareto_store(
    pareto(
        y="response_iqr",
        x="gpu_s_per_1k",
        perf_condition=dict(scenario="offline", batch_size=256),
        **IQR_AXIS,
        name="pareto_iqr_vs_gpu_s",
    ),
    name="iqr_vs_gpu_s",
)
pareto_store(
    pareto(
        y="response_iqr",
        x="flops_per_event",
        perf_condition=dict(scenario="single", batch_size=1),
        **IQR_AXIS,
        name="pareto_iqr_vs_flops",
    ),
    name="iqr_vs_flops",
)


def plots_conf(**kw):
    """A `plots` config: selection (draw/extra/skip), style, optional Pareto plot."""
    return make_config(draw="${task.plots}", extra=[], skip=[], style=STYLE, pareto=None, **kw)


plots_store(plots_conf(), name="default")
plots_store(plots_conf(hydra_defaults=["_self_", {"pareto": "iqr_vs_latency"}]), name="pareto")
