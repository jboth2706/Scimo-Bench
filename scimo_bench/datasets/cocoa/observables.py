"""COCOA observables: quantities of one set of events, computed on the truth and on every
model's output (reco). `ev` holds the events: `ev.particles.pt`, `ev.tracks.pt`, ...

Return one value per event (numpy array) or a list per event (awkward array). Add an
observable by writing a function here; it is computed and stored in every run, and can be
plotted (`plots.py`) or scored by a task metric as `truth.<name>` / `reco.<name>`.
"""

from __future__ import annotations

import numpy as np

from scimo_bench.catalogue import observable
from scimo_bench.datasets.cocoa.reader import CHARGED, NEUTRAL_HADRON, PHOTON
from scimo_bench.datasets.common import counts, energy_sum, jet

CH, NH, GAMMA = r"$h^\pm$", r"$h^0$", r"$\gamma$"

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


@observable(label=r"particle $\phi$")
def particle_phi(ev):
    return ev.particles.phi


# --- particles per class (field `cls`: charged, neutral hadron, photon) --------------------
# "charged hadrons" (h±) is the charged class: it also holds the few electrons and muons.


def of_class(objs, cls: int):
    """The objects of class `cls` (jagged)."""
    return objs[objs.cls == cls]


@observable(label=r"$N_{h^\pm}$ (charged hadrons)")
def n_charged(ev):
    return counts(of_class(ev.particles, CHARGED).pt).astype(np.float64)


@observable(label=r"$N_{h^0}$ (neutral hadrons)")
def n_neutral_hadrons(ev):
    return counts(of_class(ev.particles, NEUTRAL_HADRON).pt).astype(np.float64)


@observable(label=r"$N_\gamma$ (photons)")
def n_photons(ev):
    return counts(of_class(ev.particles, PHOTON).pt).astype(np.float64)


@observable(label=f"{CH} $p_T$", unit="GeV")
def charged_pt(ev):
    return of_class(ev.particles, CHARGED).pt


@observable(label=rf"{CH} $\eta$")
def charged_eta(ev):
    return of_class(ev.particles, CHARGED).eta


@observable(label=rf"{CH} $\phi$")
def charged_phi(ev):
    return of_class(ev.particles, CHARGED).phi


@observable(label=f"{NH} $p_T$", unit="GeV")
def neutral_hadron_pt(ev):
    return of_class(ev.particles, NEUTRAL_HADRON).pt


@observable(label=rf"{NH} $\eta$")
def neutral_hadron_eta(ev):
    return of_class(ev.particles, NEUTRAL_HADRON).eta


@observable(label=rf"{NH} $\phi$")
def neutral_hadron_phi(ev):
    return of_class(ev.particles, NEUTRAL_HADRON).phi


@observable(label=f"{GAMMA} $p_T$", unit="GeV")
def photon_pt(ev):
    return of_class(ev.particles, PHOTON).pt


@observable(label=rf"{GAMMA} $\eta$")
def photon_eta(ev):
    return of_class(ev.particles, PHOTON).eta


@observable(label=rf"{GAMMA} $\phi$")
def photon_phi(ev):
    return of_class(ev.particles, PHOTON).phi


# --- tracks (an input: truth only) ---------------------------------------------------------


@observable(label="tracks per event")
def n_tracks(ev):
    return counts(ev.tracks.pt).astype(np.float64)
