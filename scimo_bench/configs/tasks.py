"""Task configs (group `task`): what is scored on a dataset. Event list, parameters of the
comparisons, binnings, metrics, the plots drawn by default, ranking and the quality gate.
A metric names a comparison (`jet_e_response`) or an observable with its side
(`truth.jet_pt`, `reco.particle_pt`); so does its `x`.
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
        variable="jet_e_response",
        better=dict(closest_to=1.0),
        x="truth.jet_pt",
        binning=JET_PT,
        label="median jet energy response",
    ),
    builds(
        IQR,
        name="response_iqr",
        variable="jet_e_response",
        better="lower",
        x="truth.jet_pt",
        binning=JET_PT,
        label="IQR of jet energy response",
    ),
    builds(
        Efficiency,
        name="efficiency",
        variable="particle_matched",
        better="higher",
        x="truth.particle_pt",
        binning=PARTICLE_PT,
        label="particle efficiency (truth)",
    ),
    builds(
        FakeRate,
        name="fake_rate",
        variable="particle_fake",
        better="lower",
        x="reco.particle_pt",
        binning=PARTICLE_PT,
        label="particle fake rate (reco)",
    ),
    builds(
        Mean,
        name="multiplicity_diff_mean",
        variable="multiplicity_diff",
        better=dict(closest_to=0.0),
        label=r"mean $N_\mathrm{reco} - N_\mathrm{truth}$",
    ),
]

task_store(
    make_config(
        hydra_defaults=["_self_", {"/dataset@_global_.dataset": "cocoa"}],
        name="pflow",
        version=1,
        events="data/pflow_val_v1_events.txt",
        collection="particles",  # the collection the cutflow table counts
        # parameters of the comparisons (keyword arguments with the same name)
        params=dict(match_dr=0.05, emd_r=1.0),
        binnings=dict(jet_pt="${linspace:0,200,11}", particle_pt="${logspace:0,2,9}"),
        metrics=PFLOW_METRICS,
        # plots drawn by default (functions of the dataset's plots.py, or `ranking`)
        plots=[
            "jet_e_response",
            "jet_e",
            "jet_pt",
            "n_particles",
            "particle_pt",
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
