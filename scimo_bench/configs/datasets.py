"""Dataset configs (group `dataset`) and their plot catalogues (group `plot_specs`).

A dataset says how to read its truth (`truth`: a reader built with `builds`), which
collections and fields predictions must have, which module defines its variables, and
which plots exist for it. Selecting a dataset (`dataset=cocoa`, or a task's default)
also loads its plot catalogue into `plots.specs`, where every plot option can be changed:

    plots.specs.jet_pred_e.bins='${linspace:0,300,61}'  plots.specs.jet_pred_e.log_y=true
    '+plots.specs.jet_eta={_target_: scimo_bench.plotter.Distribution, variable: jet_pred_eta}'

Add a dataset with `add_dataset(...)` (here, or in a models file).
"""

from hydra_zen import builds, make_config, store

from scimo_bench.datasets.cocoa import CocoaTruth
from scimo_bench.plotter import Distribution, MetricVsX, RankingPlot

dataset_store = store(group="dataset")
specs_store = store(group="plot_specs")


def plot(cls, **kw):
    """A plot spec with every option of the plot class visible (and overridable)."""
    return builds(cls, **kw, populate_full_signature=True)


# plots that exist for every dataset (generic variables, report-based plots)
GENERIC_PLOTS = dict(
    ranking=plot(RankingPlot),
    n_pred=plot(Distribution, variable="n_pred", bins="${linspace:-0.5,30.5,32}"),
    multiplicity_diff=plot(
        Distribution, variable="multiplicity_diff", bins="${linspace:-10.5,10.5,22}", ratio=False
    ),
)


def add_dataset(
    name: str,
    truth,
    variables: str,
    collections: dict,
    plots: dict | None = None,
    event_fields=(),
    description: str = "",
) -> None:
    """Register dataset `name`: `truth` reader (a `builds(...)`), `variables` module (with a
    `REGISTRY`), required `collections` {name: [fields]}, and its plot catalogue (added to
    the generic plots)."""
    specs_store({**GENERIC_PLOTS, **(plots or {})}, name=name)
    dataset_store(
        make_config(
            hydra_defaults=["_self_", {"/plot_specs@_global_.plots.specs": name}],
            name=name,
            description=description,
            truth=truth,
            variables=variables,
            collections=collections,
            event_fields=list(event_fields),
            zen_dataclass={"cls_name": f"Dataset_{name}"},
        ),
        name=name,
    )


# relative to `paths.data`
COCOA_VAL = (
    "${paths.data}/COCOA/COCOA_samples_singlejet/val_100K_cells256/"
    "singlejet_10_100kseg_bw100.0.root"
)
COCOA_PLOTS = dict(
    jet_energy_response=plot(
        Distribution, variable="jet_energy_response", bins="${linspace:0.5,1.5,51}", ratio=False
    ),
    jet_pred_e=plot(Distribution, variable="jet_pred_e", bins="${linspace:0,250,51}"),
    jet_pred_pt=plot(Distribution, variable="jet_pred_pt", bins="${linspace:0,200,41}"),
    jet_pred_eta=plot(Distribution, variable="jet_pred_eta", bins="${linspace:-3,3,61}"),
    jet_dr=plot(Distribution, variable="jet_dr", bins="${linspace:0,0.2,41}", ratio=False),
    particle_pred_pt=plot(
        Distribution,
        variable="particle_pred_pt",
        bins="${logspace:0,2,41}",
        log_x=True,
        log_y=True,
    ),
    particle_pred_eta=plot(Distribution, variable="particle_pred_eta", bins="${linspace:-3,3,61}"),
    particle_pt_response=plot(
        Distribution, variable="particle_pt_response", bins="${linspace:0,2,51}", ratio=False
    ),
    event_emd=plot(Distribution, variable="event_emd", bins="${linspace:0,20,41}", ratio=False),
    response_vs_pt=plot(MetricVsX, metric="response_median", name="response_vs_pt"),
    response_iqr_vs_pt=plot(MetricVsX, metric="response_iqr", name="response_iqr_vs_pt"),
    efficiency_vs_pt=plot(MetricVsX, metric="efficiency", log_x=True, name="efficiency_vs_pt"),
    fake_rate_vs_pt=plot(MetricVsX, metric="fake_rate", log_x=True, name="fake_rate_vs_pt"),
)

add_dataset(
    "cocoa",
    truth=builds(CocoaTruth, path=COCOA_VAL, populate_full_signature=True),
    variables="scimo_bench.datasets.cocoa",
    collections=dict(particles=["pt", "eta", "phi", "e"]),
    plots=COCOA_PLOTS,
    description="COCOA single jet (val, 100k events), particle-flow truth from the raw ROOT file",
)
