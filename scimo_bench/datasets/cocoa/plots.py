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
    p.hist(ax, "jet_e", np.linspace(0, 250, 51), ratio_ax=rax)
    p.legend(ax)
    return fig


@plot
def jet_pt(p):
    fig, ax, rax = p.figure(ratio=True)
    p.hist(ax, "jet_pt", np.linspace(0, 200, 41), ratio_ax=rax)
    p.legend(ax)
    return fig


@plot
def jet_eta(p):
    fig, ax, rax = p.figure(ratio=True)
    p.hist(ax, "jet_eta", np.linspace(-3, 3, 61), ratio_ax=rax)
    p.legend(ax)
    return fig


@plot
def jet_e_response(p):
    fig, ax = p.figure()
    p.hist(ax, "jet_e_response", np.linspace(0.5, 1.5, 51))
    p.legend(ax)
    return fig


@plot
def jet_dr(p):
    fig, ax = p.figure()
    p.hist(ax, "jet_dr", np.linspace(0, 0.2, 41))
    p.legend(ax)
    return fig


# --- particles -----------------------------------------------------------------------------


@plot
def n_particles(p):
    fig, ax, rax = p.figure(ratio=True)
    p.hist(ax, "n_particles", np.linspace(-0.5, 30.5, 32), ratio_ax=rax)
    p.legend(ax)
    return fig


@plot
def multiplicity_diff(p):
    fig, ax = p.figure()
    p.hist(ax, "multiplicity_diff", np.linspace(-10.5, 10.5, 22))
    p.legend(ax)
    return fig


@plot
def particle_pt(p):
    fig, ax, rax = p.figure(ratio=True)
    p.hist(ax, "particle_pt", np.logspace(0, 2, 41), ratio_ax=rax)
    p.log_x(rax)
    ax.set_yscale("log")
    p.legend(ax)
    return fig


@plot
def particle_eta(p):
    fig, ax, rax = p.figure(ratio=True)
    p.hist(ax, "particle_eta", np.linspace(-3, 3, 61), ratio_ax=rax)
    p.legend(ax)
    return fig


@plot
def particle_pt_response(p):
    fig, ax = p.figure()
    p.hist(ax, "particle_pt_response", np.linspace(0, 2, 51))
    p.legend(ax)
    return fig


@plot
def event_emd(p):
    fig, ax = p.figure()
    p.hist(ax, "event_emd", np.linspace(0, 20, 41))
    p.legend(ax)
    return fig


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


@plot
def efficiency_vs_pt(p):
    fig, ax = p.figure()
    p.metric_vs_x(ax, "efficiency", log_x=True)
    p.legend(ax)
    return fig


@plot
def fake_rate_vs_pt(p):
    fig, ax = p.figure()
    p.metric_vs_x(ax, "fake_rate", log_x=True)
    p.legend(ax)
    return fig


# --- tracks (truth only) -------------------------------------------------------------------


@plot
def n_tracks(p):
    fig, ax = p.figure()
    p.hist(ax, "n_tracks", np.linspace(-0.5, 20.5, 22))
    p.legend(ax)
    return fig
