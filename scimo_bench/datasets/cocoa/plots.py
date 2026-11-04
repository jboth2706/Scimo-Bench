"""COCOA plots: one function per plot, saved as `plots/<function name>.<fmt>`.

`p` is a `scimo_bench.plotter.PlotContext`: `p.figure`, `p.hist`, `p.metric_vs_x`,
`p.legend`, `p.values(side, name)` for anything custom, and plain matplotlib on `ax`.
Which plots a run draws: the task's `plots` list, `plots.extra=[...]`, `plots.skip=[...]`.
"""

from __future__ import annotations

import numpy as np

from scimo_bench.catalogue import plot

# --- jet -----------------------------------------------------------------------------------


@plot
def jet_e(p):
    fig, ax, rax = p.figure(ratio=True)
    p.hist(ax, "jet_e", np.linspace(0, 250, 26), ratio_ax=rax)
    p.legend(ax)
    return fig


@plot
def jet_pt(p):
    fig, ax, rax = p.figure(ratio=True)
    p.hist(ax, "jet_pt", np.linspace(0, 200, 21), ratio_ax=rax)
    p.legend(ax)
    return fig


@plot
def jet_eta(p):
    fig, ax, rax = p.figure(ratio=True)
    p.hist(ax, "jet_eta", np.linspace(-3, 3, 31), ratio_ax=rax)
    p.legend(ax)
    return fig


@plot
def jet_e_response(p):
    fig, ax = p.figure()
    p.hist(ax, "jet_e_response", np.linspace(0.5, 1.5, 26))
    p.legend(ax)
    return fig


@plot
def jet_dr(p):
    fig, ax = p.figure()
    p.hist(ax, "jet_dr", np.linspace(0, 0.1, 16))
    p.legend(ax)
    return fig


# --- particles -----------------------------------------------------------------------------


@plot
def n_particles(p):
    fig, ax, rax = p.figure(ratio=True)
    p.hist(ax, "n_particles", np.arange(-0.5, 32, 1), ratio_ax=rax)
    p.legend(ax)
    return fig


@plot
def multiplicity_diff(p):
    fig, ax = p.figure()
    p.hist(ax, "multiplicity_diff", np.arange(-10.5, 11, 1))
    p.legend(ax)
    return fig


@plot
def particle_pt(p):
    fig, ax, rax = p.figure(ratio=True)
    p.hist(ax, "particle_pt", np.linspace(0, 100, 26), ratio_ax=rax)
    p.legend(ax)
    return fig


@plot
def particle_eta(p):
    fig, ax, rax = p.figure(ratio=True)
    p.hist(ax, "particle_eta", np.linspace(-3, 3, 31), ratio_ax=rax)
    p.legend(ax)
    return fig


@plot
def particle_phi(p):
    fig, ax, rax = p.figure(ratio=True)
    p.hist(ax, "particle_phi", np.linspace(-np.pi, np.pi, 31), ratio_ax=rax)
    p.legend(ax)
    return fig


# --- particles per class: charged hadrons, neutral hadrons, photons ------------------------

PT_BINS = np.linspace(0, 100, 26)
ETA_BINS = np.linspace(-3, 3, 31)
PHI_BINS = np.linspace(-np.pi, np.pi, 31)


def ratio_hist(p, name, bins):
    """Truth and models with a ratio panel; the legend above."""
    fig, ax, rax = p.figure(ratio=True)
    p.hist(ax, name, bins, ratio_ax=rax)
    p.legend(ax)
    return fig


def single_hist(p, name, bins):
    """A comparison: one line per model."""
    fig, ax = p.figure()
    p.hist(ax, name, bins)
    p.legend(ax)
    return fig


@plot
def n_charged(p):
    return ratio_hist(p, "n_charged", np.arange(-0.5, 21, 1))


@plot
def n_neutral_hadrons(p):
    return ratio_hist(p, "n_neutral_hadrons", np.arange(-0.5, 11, 1))


@plot
def n_photons(p):
    return ratio_hist(p, "n_photons", np.arange(-0.5, 21, 1))


@plot
def n_charged_diff(p):
    return single_hist(p, "n_charged_diff", np.arange(-6.5, 7, 1))


@plot
def n_neutral_hadrons_diff(p):
    return single_hist(p, "n_neutral_hadrons_diff", np.arange(-6.5, 7, 1))


@plot
def n_photons_diff(p):
    return single_hist(p, "n_photons_diff", np.arange(-6.5, 7, 1))


@plot
def charged_pt(p):
    return ratio_hist(p, "charged_pt", PT_BINS)


@plot
def charged_eta(p):
    return ratio_hist(p, "charged_eta", ETA_BINS)


@plot
def charged_phi(p):
    return ratio_hist(p, "charged_phi", PHI_BINS)


@plot
def neutral_hadron_pt(p):
    return ratio_hist(p, "neutral_hadron_pt", PT_BINS)


@plot
def neutral_hadron_eta(p):
    return ratio_hist(p, "neutral_hadron_eta", ETA_BINS)


@plot
def neutral_hadron_phi(p):
    return ratio_hist(p, "neutral_hadron_phi", PHI_BINS)


@plot
def photon_pt(p):
    return ratio_hist(p, "photon_pt", np.linspace(0, 50, 26))


@plot
def photon_eta(p):
    return ratio_hist(p, "photon_eta", ETA_BINS)


@plot
def photon_phi(p):
    return ratio_hist(p, "photon_phi", PHI_BINS)


# --- jet direction and mass; energy per class ----------------------------------------------

DIFF_BINS = np.linspace(-0.05, 0.05, 41)
RESPONSE_BINS = np.linspace(0, 2, 41)


@plot
def jet_deta(p):
    return single_hist(p, "jet_deta", DIFF_BINS)


@plot
def jet_dphi(p):
    return single_hist(p, "jet_dphi", DIFF_BINS)


@plot
def jet_m_response(p):
    return single_hist(p, "jet_m_response", RESPONSE_BINS)


@plot
def e_charged_response(p):
    return single_hist(p, "e_charged_response", RESPONSE_BINS)


@plot
def e_neutral_hadrons_response(p):
    return single_hist(p, "e_neutral_hadrons_response", RESPONSE_BINS)


@plot
def e_photons_response(p):
    return single_hist(p, "e_photons_response", RESPONSE_BINS)


# --- metrics in bins (from report.csv): median above, IQR below ----------------------------


def median_iqr(p, median, iqr, median_ylim=None, iqr_ylim=None):
    """Two binned metrics on a shared x axis: `median` (top) and `iqr` (bottom).
    `median_ylim`/`iqr_ylim`: y ranges [low, high] of the two panels."""
    fig, (top, bottom) = p.figure(panels=2)
    p.metric_vs_x(top, median)
    p.metric_vs_x(bottom, iqr)
    if median_ylim is not None:
        top.set_ylim(*median_ylim)
    if iqr_ylim is not None:
        bottom.set_ylim(*iqr_ylim)
    top.set_xlabel("")
    p.legend(top)
    return fig


@plot
def response_vs_e(p):
    return median_iqr(
        p, "response_median", "response_iqr", median_ylim=(0.9, 1.1), iqr_ylim=(0.02, 0.15)
    )


@plot
def response_vs_n(p):
    return median_iqr(p, "response_median_vs_n", "response_iqr_vs_n")


@plot
def response_vs_charged_frac(p):
    return median_iqr(p, "response_median_vs_charged_frac", "response_iqr_vs_charged_frac")


@plot
def response_vs_nh_frac(p):
    return median_iqr(p, "response_median_vs_nh_frac", "response_iqr_vs_nh_frac")


@plot
def response_vs_ntracks(p):
    return median_iqr(p, "response_median_vs_ntracks", "response_iqr_vs_ntracks")


@plot
def response_vs_track_width(p):
    return median_iqr(p, "response_median_vs_track_width", "response_iqr_vs_track_width")


@plot
def deta_vs_eta(p):
    return median_iqr(p, "deta_median", "deta_iqr")


@plot
def deta_vs_e(p):
    return median_iqr(p, "deta_median_vs_e", "deta_iqr_vs_e")


@plot
def dphi_vs_e(p):
    return median_iqr(p, "dphi_median", "dphi_iqr")


# --- tracks (truth only) -------------------------------------------------------------------


@plot
def n_tracks(p):
    fig, ax = p.figure()
    p.hist(ax, "n_tracks", np.arange(-0.5, 22, 1))
    p.legend(ax)
    return fig
