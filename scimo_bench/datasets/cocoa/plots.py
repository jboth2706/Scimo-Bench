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
    p.hist(ax, "jet_dr", np.linspace(0, 0.2, 21))
    p.legend(ax)
    return fig


# --- particles -----------------------------------------------------------------------------


@plot
def n_particles(p):
    fig, ax, rax = p.figure(ratio=True)
    p.hist(ax, "n_particles", np.arange(-0.5, 32, 2), ratio_ax=rax)
    p.legend(ax)
    return fig


@plot
def multiplicity_diff(p):
    fig, ax = p.figure()
    p.hist(ax, "multiplicity_diff", np.arange(-10.5, 12, 2))
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
    return ratio_hist(p, "photon_pt", PT_BINS)


@plot
def photon_eta(p):
    return ratio_hist(p, "photon_eta", ETA_BINS)


@plot
def photon_phi(p):
    return ratio_hist(p, "photon_phi", PHI_BINS)


# --- metrics in bins (from report.csv) -----------------------------------------------------


@plot
def response_vs_pt(p):
    fig, ax = p.figure()
    p.metric_vs_x(ax, "response_median")
    p.legend(ax)
    return fig


@plot
def response_iqr_vs_pt(p):
    fig, ax = p.figure()
    p.metric_vs_x(ax, "response_iqr")
    p.legend(ax)
    return fig


# --- tracks (truth only) -------------------------------------------------------------------


@plot
def n_tracks(p):
    fig, ax = p.figure()
    p.hist(ax, "n_tracks", np.arange(-0.5, 22, 2))
    p.legend(ax)
    return fig
