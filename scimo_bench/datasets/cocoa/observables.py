"""COCOA observables: quantities of one set of events, computed on the truth and on every
model's output (reco). `ev` holds the events: `ev.particles.pt`, `ev.tracks.pt`, ...

Return one value per event (numpy array) or a list per event (awkward array). Add an
observable by writing a function here; it is computed and stored in every run, and can be
plotted (`plots.py`) or scored by a task metric as `truth.<name>` / `reco.<name>`.
"""

from __future__ import annotations

import numpy as np

from scimo_bench.catalogue import observable
from scimo_bench.datasets.common import counts, energy_sum, jet

# --- jet: the vector sum of all particles of the event -------------------------------------


@observable(label="jet $p_T$", unit="GeV")
def jet_pt(ev):
    return jet(ev.particles)["pt"]


@observable(label=r"jet $\eta$")
def jet_eta(ev):
    return jet(ev.particles)["eta"]


@observable(label=r"jet $\phi$")
def jet_phi(ev):
    return jet(ev.particles)["phi"]


@observable(label="jet $E$", unit="GeV")
def jet_e(ev):
    return energy_sum(ev.particles)


# --- particles -----------------------------------------------------------------------------


@observable(label="particles per event")
def n_particles(ev):
    return counts(ev.particles.pt).astype(np.float64)


@observable(label="particle $p_T$", unit="GeV")
def particle_pt(ev):
    return ev.particles.pt


@observable(label=r"particle $\eta$")
def particle_eta(ev):
    return ev.particles.eta


# --- tracks (an input: truth only) ---------------------------------------------------------


@observable(label="tracks per event")
def n_tracks(ev):
    return counts(ev.tracks.pt).astype(np.float64)
