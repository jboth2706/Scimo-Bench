"""Task configs (group `task`): what is scored on a dataset. Event list, parameters of the
comparisons, binnings, metrics, the plots drawn by default, ranking and the quality gate.
A metric names a comparison (`jet_e_response`) or an observable with its side
(`truth.jet_pt`, `reco.particle_pt`); so does its `x`.
A task names its default dataset (`dataset=` overrides it). The event list is a file in
this repo, versioned with the task."""

from hydra_zen import builds, make_config, store

from scimo_bench.scorer import IQR, Mean, Median, Std

task_store = store(group="task")


def binning(name: str) -> str:
    """The task's binning `name` (`task.binnings`)."""
    return "${task.binnings." + name + "}"


def median_iqr(name, variable, ideal, label, x=None, bins=None, suffix="", rank=True):
    """`<name>_median<suffix>` and `<name>_iqr<suffix>` of `variable`, binned in `x`."""
    binned = dict(x=x, binning=binning(bins)) if x else {}
    return [
        builds(
            Median,
            name=f"{name}_median{suffix}",
            variable=variable,
            better=dict(closest_to=ideal),
            label=f"median {label}",
            rank=rank,
            **binned,
        ),
        builds(
            IQR,
            name=f"{name}_iqr{suffix}",
            variable=variable,
            better="lower",
            label=f"IQR of {label}",
            rank=rank,
            **binned,
        ),
    ]


def mean_std(variable, ideal, label, rank=True):
    """`<variable>_mean` and `<variable>_std`."""
    return [
        builds(
            Mean,
            name=f"{variable}_mean",
            variable=variable,
            better=dict(closest_to=ideal),
            label=f"mean {label}",
            rank=rank,
        ),
        builds(
            Std,
            name=f"{variable}_std",
            variable=variable,
            better="lower",
            label=f"std of {label}",
            rank=rank,
        ),
    ]


E_RESPONSE = "jet energy response"
# binned variants of the jet energy response: (suffix, x, binning)
E_RESPONSE_VS = [
    ("_vs_n", "truth.n_particles", "n_particles"),
    ("_vs_charged_frac", "truth.charged_e_fraction", "e_fraction"),
    ("_vs_nh_frac", "truth.neutral_hadron_e_fraction", "e_fraction"),
    ("_vs_ntracks", "truth.n_tracks", "n_tracks"),
    ("_vs_track_width", "truth.track_width", "track_width"),
]
CLASS_E_RESPONSES = [
    ("e_charged_response", r"$E_{h^\pm,\mathrm{reco}} / E_{h^\pm,\mathrm{truth}}$"),
    ("e_neutral_hadrons_response", r"$E_{h^0,\mathrm{reco}} / E_{h^0,\mathrm{truth}}$"),
    ("e_photons_response", r"$E_{\gamma,\mathrm{reco}} / E_{\gamma,\mathrm{truth}}$"),
]
MULTIPLICITY_DIFFS = [
    ("multiplicity_diff", r"$N_\mathrm{reco} - N_\mathrm{truth}$"),
    ("n_charged_diff", r"$N_{h^\pm,\mathrm{reco}} - N_{h^\pm,\mathrm{truth}}$"),
    ("n_neutral_hadrons_diff", r"$N_{h^0,\mathrm{reco}} - N_{h^0,\mathrm{truth}}$"),
    ("n_photons_diff", r"$N_{\gamma,\mathrm{reco}} - N_{\gamma,\mathrm{truth}}$"),
]

# A base name (`response_iqr`) is ranked; its `_vs_<x>` variants only reported.
PFLOW_METRICS = [
    *median_iqr("response", "jet_e_response", 1.0, E_RESPONSE, "truth.jet_e", "jet_e"),
    *[
        m
        for suffix, x, bins in E_RESPONSE_VS
        for m in median_iqr("response", "jet_e_response", 1.0, E_RESPONSE, x, bins, suffix, False)
    ],
    *median_iqr("deta", "jet_deta", 0.0, r"jet $\Delta\eta$", "truth.jet_eta", "jet_eta"),
    *median_iqr(
        "deta", "jet_deta", 0.0, r"jet $\Delta\eta$", "truth.jet_e", "jet_e", "_vs_e", False
    ),
    *median_iqr("dphi", "jet_dphi", 0.0, r"jet $\Delta\phi$", "truth.jet_e", "jet_e"),
    # no jet mass metric yet: the reco mass is 0 for ~40 % of nanoHEP jets (e < |p|)
    *[m for name, label in CLASS_E_RESPONSES for m in mean_std(name, 1.0, label)],
    *[
        m
        for name, label in CLASS_E_RESPONSES
        for m in median_iqr(name, name, 1.0, label, rank=False)
    ],
    *[m for name, label in MULTIPLICITY_DIFFS for m in mean_std(name, 0.0, label)],
]

task_store(
    make_config(
        hydra_defaults=["_self_", {"/dataset@_global_.dataset": "cocoa"}],
        name="pflow",
        version=1,
        events="data/pflow_val_v1_events.txt",
        collection="particles",  # the collection the cutflow table counts
        # parameters of the comparisons (keyword arguments with the same name); none needed
        params={},
        binnings=dict(
            jet_e="${linspace:20,200,10}",
            jet_pt="${linspace:0,200,11}",
            jet_eta="${linspace:-2.5,2.5,11}",
            n_particles=[*[n + 0.5 for n in range(16)], 27.5],  # 1, 2, ..., 15, 16-27
            # last edge above 1: 12 % of the events are all charged (fraction exactly 1)
            e_fraction=[*[i / 10 for i in range(10)], 1.001],
            n_tracks=[*[n - 0.5 for n in range(11)], 13.5],  # 0, 1, ..., 9, 10-13
            # median 0.046, 99 % below 0.32, max 0.77 (COCOA val)
            track_width=[0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.8],
        ),
        metrics=PFLOW_METRICS,
        # plots drawn by default (functions of the dataset's plots.py, or `ranking`)
        plots=[
            "jet_e_response",
            "jet_e",
            "jet_pt",
            "n_particles",
            "particle_pt",
            "n_charged",
            "n_neutral_hadrons",
            "n_photons",
            "n_charged_diff",
            "n_neutral_hadrons_diff",
            "n_photons_diff",
            *[
                f"{c}_{v}"
                for c in ("charged", "neutral_hadron", "photon")
                for v in ("pt", "eta", "phi")
            ],
            "jet_dr",
            "jet_deta",
            "jet_dphi",
            "jet_m_response",
            "e_charged_response",
            "e_neutral_hadrons_response",
            "e_photons_response",
            "multiplicity_diff",
            "response_vs_e",
            "response_vs_n",
            "response_vs_charged_frac",
            "response_vs_nh_frac",
            "response_vs_ntracks",
            "response_vs_track_width",
            "deta_vs_eta",
            "deta_vs_e",
            "dphi_vs_e",
            "ranking",
        ],
        ranking=dict(primary="response_iqr"),
        normalization=dict(baseline=None, reference=None),
        quality_gate=dict(metric="response_iqr", max=0.25),
    ),
    name="pflow",
)
