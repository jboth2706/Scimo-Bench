"""Task configs (group `task`): what is scored on a dataset. Event list, variable
parameters, binnings, metrics, the plots drawn by default, ranking and the quality gate.
A task names its default dataset (`dataset=` overrides it). The event list is a file in
this repo, versioned with the task."""

from hydra_zen import builds, make_config, store

from scimo_bench.scorer import IQR, Efficiency, FakeRate, Mean, Median

task_store = store(group="task")

JET_PT = "${task.binnings.jet_pt}"
PARTICLE_PT = "${task.binnings.particle_pt}"

PFLOW_METRICS = [
    builds(
        Median, name="emd_median", variable="event_emd", better="lower", label="median EMD [GeV]"
    ),
    builds(
        Median,
        name="response_median",
        variable="jet_energy_response",
        better=dict(closest_to=1.0),
        x="jet_truth_pt",
        binning=JET_PT,
        label="median jet energy response",
    ),
    builds(
        IQR,
        name="response_iqr",
        variable="jet_energy_response",
        better="lower",
        x="jet_truth_pt",
        binning=JET_PT,
        label="IQR of jet energy response",
    ),
    builds(
        Efficiency,
        name="efficiency",
        variable="particle_matched",
        better="higher",
        x="particle_truth_pt",
        binning=PARTICLE_PT,
        label="particle efficiency",
    ),
    builds(
        FakeRate,
        name="fake_rate",
        variable="particle_fake",
        better="lower",
        x="particle_pred_pt",
        binning=PARTICLE_PT,
        label="particle fake rate",
    ),
    builds(
        Mean,
        name="multiplicity_diff_mean",
        variable="multiplicity_diff",
        better=dict(closest_to=0.0),
        label=r"mean $N_\mathrm{pred} - N_\mathrm{truth}$",
    ),
]

task_store(
    make_config(
        hydra_defaults=["_self_", {"/dataset@_global_.dataset": "cocoa"}],
        name="pflow",
        version=1,
        events="data/pflow_val_v1_events.txt",
        # passed to the variables (ctx.param): collection scored, matching ΔR, EMD ΔR cap
        params=dict(collection="particles", match_dr=0.05, emd_r=1.0),
        binnings=dict(jet_pt="${linspace:0,200,11}", particle_pt="${logspace:0,2,9}"),
        metrics=PFLOW_METRICS,
        # extra variables stored in per_event.parquet (for plots / later analysis)
        variables=["jet_dr", "particle_pt_response", "particle_pred_eta"],
        # plots drawn by default (names from the dataset's catalogue `plots.specs`, or any
        # variable, drawn as a distribution)
        plots=[
            "jet_energy_response",
            "jet_pred_e",
            "jet_pred_pt",
            "n_pred",
            "particle_pred_pt",
            "event_emd",
            "response_vs_pt",
            "response_iqr_vs_pt",
            "efficiency_vs_pt",
            "fake_rate_vs_pt",
            "ranking",
        ],
        ranking=dict(primary="response_iqr"),
        normalization=dict(baseline=None, reference=None),
        quality_gate=dict(metric="response_iqr", max=0.25),
    ),
    name="pflow",
)
