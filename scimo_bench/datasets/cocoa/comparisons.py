"""COCOA comparisons: quantities of the truth and one model's output (reco), row by row.

Task parameters would be keyword arguments with defaults (`task.params` overrides them);
these need none. Labels say which side each part comes from.
"""

from __future__ import annotations

import numpy as np

from scimo_bench.catalogue import comparison
from scimo_bench.datasets.cocoa.observables import (
    e_charged,
    e_neutral_hadrons,
    e_photons,
    jet_e,
    jet_m,
    n_charged,
    n_neutral_hadrons,
    n_particles,
    n_photons,
)
from scimo_bench.datasets.common import delta_phi, jet, ratio

# --- jet -----------------------------------------------------------------------------------


@comparison(label=r"$E_\mathrm{reco} / E_\mathrm{truth}$ (jet)", reference=1.0)
def jet_e_response(truth, reco):
    return ratio(jet_e(reco), jet_e(truth))


@comparison(label=r"$p_{T,\mathrm{reco}} / p_{T,\mathrm{truth}}$ (jet)", reference=1.0)
def jet_pt_response(truth, reco):
    return ratio(jet(reco.particles)["pt"], jet(truth.particles)["pt"])


@comparison(label=r"$\Delta R$(reco jet, truth jet)", reference=0.0)
def jet_dr(truth, reco):
    t, r = jet(truth.particles), jet(reco.particles)
    return np.hypot(r["eta"] - t["eta"], delta_phi(r["phi"], t["phi"]))


@comparison(label=r"$\eta_\mathrm{reco} - \eta_\mathrm{truth}$ (jet)", reference=0.0)
def jet_deta(truth, reco):
    return jet(reco.particles)["eta"] - jet(truth.particles)["eta"]


@comparison(label=r"$\phi_\mathrm{reco} - \phi_\mathrm{truth}$ (jet)", unit="rad", reference=0.0)
def jet_dphi(truth, reco):
    return delta_phi(jet(reco.particles)["phi"], jet(truth.particles)["phi"])


@comparison(label=r"$m_\mathrm{reco} / m_\mathrm{truth}$ (jet)", reference=1.0)
def jet_m_response(truth, reco):
    return ratio(jet_m(reco), jet_m(truth))


# --- energy per class ----------------------------------------------------------------------


@comparison(label=r"$E_{h^\pm,\mathrm{reco}} / E_{h^\pm,\mathrm{truth}}$", reference=1.0)
def e_charged_response(truth, reco):
    return ratio(e_charged(reco), e_charged(truth))


@comparison(label=r"$E_{h^0,\mathrm{reco}} / E_{h^0,\mathrm{truth}}$", reference=1.0)
def e_neutral_hadrons_response(truth, reco):
    return ratio(e_neutral_hadrons(reco), e_neutral_hadrons(truth))


@comparison(label=r"$E_{\gamma,\mathrm{reco}} / E_{\gamma,\mathrm{truth}}$", reference=1.0)
def e_photons_response(truth, reco):
    return ratio(e_photons(reco), e_photons(truth))


@comparison(label=r"$N_\mathrm{reco} - N_\mathrm{truth}$ (particles)", reference=0.0)
def multiplicity_diff(truth, reco):
    return n_particles(reco) - n_particles(truth)


@comparison(label=r"$N_{h^\pm,\mathrm{reco}} - N_{h^\pm,\mathrm{truth}}$", reference=0.0)
def n_charged_diff(truth, reco):
    return n_charged(reco) - n_charged(truth)


@comparison(label=r"$N_{h^0,\mathrm{reco}} - N_{h^0,\mathrm{truth}}$", reference=0.0)
def n_neutral_hadrons_diff(truth, reco):
    return n_neutral_hadrons(reco) - n_neutral_hadrons(truth)


@comparison(label=r"$N_{\gamma,\mathrm{reco}} - N_{\gamma,\mathrm{truth}}$", reference=0.0)
def n_photons_diff(truth, reco):
    return n_photons(reco) - n_photons(truth)
