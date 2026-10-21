"""COCOA comparisons: quantities of the truth and one model's output (reco), row by row.

Task parameters are keyword arguments with defaults; `task.params` overrides them
(`match_dr`, `emd_r`). Labels say which side each part comes from.
"""

from __future__ import annotations

import numpy as np

from scimo_bench.catalogue import comparison
from scimo_bench.datasets.cocoa.observables import jet_e, n_particles
from scimo_bench.datasets.common import delta_phi, emd, flat, jet, match, ratio, unflatten

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


@comparison(label=r"$N_\mathrm{reco} - N_\mathrm{truth}$ (particles)", reference=0.0)
def multiplicity_diff(truth, reco):
    return n_particles(reco) - n_particles(truth)


# --- particles, matched in ΔR --------------------------------------------------------------


@comparison(label="truth particle matched to reco", reference=1.0)
def particle_matched(truth, reco, match_dr=0.05):
    """1 per truth particle with a matched reco particle, else 0 (→ efficiency)."""
    m = match(truth, reco, match_dr)
    return unflatten((m["truth_idx"] >= 0).astype(np.float64), m["truth_counts"])


@comparison(label="reco particle without truth match", reference=0.0)
def particle_fake(truth, reco, match_dr=0.05):
    """1 per reco particle without a truth match, else 0 (→ fake rate)."""
    m = match(truth, reco, match_dr)
    return unflatten((~m["reco_matched"]).astype(np.float64), m["reco_counts"])


@comparison(label=r"$\Delta R$(reco, truth) of matched particles", reference=0.0)
def particle_dr(truth, reco, match_dr=0.05):
    """Per truth particle (NaN if unmatched)."""
    m = match(truth, reco, match_dr)
    return unflatten(m["truth_dr"], m["truth_counts"])


@comparison(
    label=r"$p_{T,\mathrm{reco}} / p_{T,\mathrm{truth}}$ (matched particles)", reference=1.0
)
def particle_pt_response(truth, reco, match_dr=0.05):
    """Per truth particle (NaN if unmatched)."""
    m = match(truth, reco, match_dr)
    idx = m["truth_idx"]
    r_pt, t_pt = flat(reco.particles.pt), flat(truth.particles.pt)
    resp = np.full(len(idx), np.nan)
    ok = idx >= 0
    resp[ok] = r_pt[idx[ok]] / t_pt[ok]
    return unflatten(resp, m["truth_counts"])


# --- event distance ------------------------------------------------------------------------


@comparison(label="EMD(truth, reco)", unit="GeV", reference=0.0)
def event_emd(truth, reco, emd_r=1.0):
    """Unbalanced EMD between truth and reco particles per event (`emd_r` caps ΔR)."""
    return emd(truth.particles, reco.particles, emd_r)
