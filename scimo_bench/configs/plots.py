"""Plot configs (group `plots`): which plots are drawn and how they look.

- `plots.specs`: the plot catalogue of the selected dataset (`configs/datasets.py`); every
  option of every plot can be overridden, e.g. `plots.specs.jet_pred_e.log_y=true`.
- `plots.draw`: names drawn (default: the task's `plots`); `plots.extra` adds names,
  `plots.skip` removes some. A name is a catalogue entry, a registered plot or a variable
  (drawn as a distribution).
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
    grid_color="#e4e3df",
    font_size=10,
    line_width=2.0,
    figsize=[6.0, 4.0],
    ratio_figsize=[6.0, 5.0],
    dpi=150,
    legend="above",  # above the axes, or any matplotlib `loc` (e.g. "best", "upper right")
    rc={},  # any matplotlib rcParams, applied last
)


def pareto(**kw):
    """A Pareto spec with every option visible."""
    return builds(ParetoPlot, **kw, populate_full_signature=True)


pareto_store(
    pareto(
        y="emd_median",
        x="latency_p50",
        perf_condition=dict(scenario="single", batch_size=1),
        name="pareto_emd_vs_latency",
    ),
    name="emd_vs_latency",
)
pareto_store(
    pareto(
        y="emd_median",
        x="gpu_s_per_1k",
        perf_condition=dict(scenario="offline", batch_size=256),
        name="pareto_emd_vs_gpu_s",
    ),
    name="emd_vs_gpu_s",
)
pareto_store(
    pareto(
        y="emd_median",
        x="flops_per_event",
        perf_condition=dict(scenario="single", batch_size=1),
        name="pareto_emd_vs_flops",
    ),
    name="emd_vs_flops",
)


def plots_conf(**kw):
    """A `plots` config: selection (draw/extra/skip), style, optional Pareto plot."""
    return make_config(draw="${task.plots}", extra=[], skip=[], style=STYLE, pareto=None, **kw)


plots_store(plots_conf(), name="default")
plots_store(plots_conf(hydra_defaults=["_self_", {"pareto": "emd_vs_latency"}]), name="pareto")
