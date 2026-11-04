"""COCOA observables: quantities of one set of events, computed on the truth and on every
model's output (reco). `ev` holds the events: `ev.particles.pt`, `ev.tracks.pt`, ...

Return one value per event (numpy array) or a list per event (awkward array). Add an
observable by writing a function here; it is computed and stored in every run, and can be
plotted (`plots.py`) or scored by a task metric as `truth.<name>` / `reco.<name>`.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

from scimo_bench.catalogue import observable
from scimo_bench.datasets.cocoa.reader import CHARGED, NEUTRAL_HADRON, PHOTON
from scimo_bench.datasets.common import counts, delta_phi, energy_sum, f64, jet, jet_mass, ratio

CH, NH, GAMMA = "charged hadrons", "neutral hadrons", "photons"

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


@observable(label="jet mass", unit="GeV")
def jet_m(ev):
    return jet_mass(ev.particles)


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


@observable(label=f"$p_T$ ({CH})", unit="GeV")
def charged_pt(ev):
    return of_class(ev.particles, CHARGED).pt


@observable(label=rf"$\eta$ ({CH})")
def charged_eta(ev):
    return of_class(ev.particles, CHARGED).eta


@observable(label=rf"$\phi$ ({CH})")
def charged_phi(ev):
    return of_class(ev.particles, CHARGED).phi


@observable(label=f"$p_T$ ({NH})", unit="GeV")
def neutral_hadron_pt(ev):
    return of_class(ev.particles, NEUTRAL_HADRON).pt


@observable(label=rf"$\eta$ ({NH})")
def neutral_hadron_eta(ev):
    return of_class(ev.particles, NEUTRAL_HADRON).eta


@observable(label=rf"$\phi$ ({NH})")
def neutral_hadron_phi(ev):
    return of_class(ev.particles, NEUTRAL_HADRON).phi


@observable(label=f"$p_T$ ({GAMMA})", unit="GeV")
def photon_pt(ev):
    return of_class(ev.particles, PHOTON).pt


@observable(label=rf"$\eta$ ({GAMMA})")
def photon_eta(ev):
    return of_class(ev.particles, PHOTON).eta


@observable(label=rf"$\phi$ ({GAMMA})")
def photon_phi(ev):
    return of_class(ev.particles, PHOTON).phi


# --- energy per class ----------------------------------------------------------------------


@observable(label=f"$E$ ({CH})", unit="GeV")
def e_charged(ev):
    return energy_sum(of_class(ev.particles, CHARGED))


@observable(label=f"$E$ ({NH})", unit="GeV")
def e_neutral_hadrons(ev):
    return energy_sum(of_class(ev.particles, NEUTRAL_HADRON))


@observable(label=f"$E$ ({GAMMA})", unit="GeV")
def e_photons(ev):
    return energy_sum(of_class(ev.particles, PHOTON))


@observable(label=r"charged energy fraction $E_{h^\pm} / E_\mathrm{jet}$")
def charged_e_fraction(ev):
    return ratio(e_charged(ev), jet_e(ev))


@observable(label=r"neutral hadron energy fraction $E_{h^0} / E_\mathrm{jet}$")
def neutral_hadron_e_fraction(ev):
    return ratio(e_neutral_hadrons(ev), jet_e(ev))


# --- tracks (an input: truth only) ---------------------------------------------------------


@observable(label="tracks per event")
def n_tracks(ev):
    return counts(ev.tracks.pt).astype(np.float64)


@observable(label=r"track width ($p_T$-weighted $\Delta R$ to the jet axis)")
def track_width(ev):
    axis = jet(ev.particles)
    tr = ev.tracks
    pt = f64(tr.pt)
    dr = np.hypot(f64(tr.eta) - axis["eta"], delta_phi(f64(tr.phi), axis["phi"]))
    return ratio(ak.to_numpy(ak.sum(pt * dr, axis=1)), ak.to_numpy(ak.sum(pt, axis=1)))
