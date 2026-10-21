"""Physics helpers shared by datasets: jets, ΔR, matching, EMD, and array conversions.

Inputs are awkward arrays of objects (jagged: a list per event) with fields such as `pt`,
`eta`, `phi`; outputs are numpy arrays (one value per event) or jagged arrays.
"""

from __future__ import annotations

import awkward as ak
import numpy as np
import ot

# --- array conversions ---------------------------------------------------------------------


def flat(x: ak.Array) -> np.ndarray:
    """Jagged → flat float64 numpy array."""
    return ak.to_numpy(ak.flatten(x, axis=1)).astype(np.float64)


def f64(x: ak.Array) -> ak.Array:
    """Jagged array as float64 (sums in float64 match numpy references)."""
    return ak.values_astype(x, np.float64)


def counts(x: ak.Array) -> np.ndarray:
    """Objects per event."""
    return ak.to_numpy(ak.num(x, axis=1))


def offsets(c: np.ndarray) -> np.ndarray:
    """Counts → start offsets into the flat arrays (length n + 1)."""
    return np.concatenate([[0], np.cumsum(c)])


def unflatten(values: np.ndarray, c: np.ndarray) -> ak.Array:
    """Flat values + counts → jagged array."""
    return ak.unflatten(values, c)


def ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num / den, inf/NaN where den is 0 (dropped later as non-finite)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den


# --- kinematics ----------------------------------------------------------------------------


def delta_phi(a, b):
    """φ difference wrapped to (−π, π]."""
    d = a - b
    return np.arctan2(np.sin(d), np.cos(d))


def jet(objs: ak.Array) -> dict[str, np.ndarray]:
    """Massless vector sum of all objects of each event: `pt`, `eta`, `phi` per event."""
    pt, eta, phi = f64(objs.pt), f64(objs.eta), f64(objs.phi)
    px = ak.to_numpy(ak.sum(pt * np.cos(phi), axis=1))
    py = ak.to_numpy(ak.sum(pt * np.sin(phi), axis=1))
    pz = ak.to_numpy(ak.sum(pt * np.sinh(eta), axis=1))
    jpt = np.hypot(px, py)
    safe = np.where(jpt == 0, 1e-10, jpt)
    return dict(pt=jpt, eta=np.arcsinh(pz / safe), phi=np.arctan2(py, px))


def energy_sum(objs: ak.Array) -> np.ndarray:
    """Sum of the objects' energies per event."""
    return ak.to_numpy(ak.sum(f64(objs.e), axis=1))


# --- matching ------------------------------------------------------------------------------

# the last matching: (truth, reco, collection, max_dr, result). The comparisons of one
# model get the same truth / reco arrays, so the matching runs once per model.
_last_match: tuple | None = None


def match(
    truth: ak.Array, reco: ak.Array, max_dr: float, collection: str = "particles"
) -> dict[str, np.ndarray]:
    """Greedy ΔR matching of `collection`: truth objects by descending pT take the nearest
    free reco object within `max_dr`. Flat arrays: `truth_idx` (matched reco object, flat
    index or -1), `truth_dr` (NaN if unmatched), `reco_matched`; plus both count arrays."""
    global _last_match
    if _last_match is not None:
        t, r, coll, dr_max, out = _last_match
        if t is truth and r is reco and coll == collection and dr_max == max_dr:
            return out
    t_objs, r_objs = truth[collection], reco[collection]
    tc, rc = counts(t_objs.pt), counts(r_objs.eta)
    to, ro = offsets(tc), offsets(rc)
    tpt, teta, tphi = flat(t_objs.pt), flat(t_objs.eta), flat(t_objs.phi)
    reta, rphi = flat(r_objs.eta), flat(r_objs.phi)
    truth_idx = np.full(len(tpt), -1, dtype=np.int64)
    truth_dr = np.full(len(tpt), np.nan)
    reco_matched = np.zeros(len(reta), dtype=bool)
    for i in np.flatnonzero((tc > 0) & (rc > 0)):
        a, b, c, d = to[i], to[i + 1], ro[i], ro[i + 1]
        dr = np.hypot(
            teta[a:b, None] - reta[None, c:d], delta_phi(tphi[a:b, None], rphi[None, c:d])
        )
        for k in np.argsort(-tpt[a:b], kind="stable"):
            row = np.where(reco_matched[c:d], np.inf, dr[k])
            j = int(np.argmin(row))
            if row[j] < max_dr:
                reco_matched[c + j] = True
                truth_idx[a + k] = c + j
                truth_dr[a + k] = row[j]
    out = dict(
        truth_idx=truth_idx,
        truth_dr=truth_dr,
        reco_matched=reco_matched,
        truth_counts=tc,
        reco_counts=rc,
    )
    _last_match = (truth, reco, collection, max_dr, out)
    return out


# --- event distance ------------------------------------------------------------------------


def emd_event(t_pt, t_eta, t_phi, r_pt, r_eta, r_phi, R: float = 1.0) -> float:
    """Unbalanced EMD [GeV] between two sets of objects in one event (port of
    `pflow_report._compute_emd_event`).

    Cost per unit pT is ΔR capped at R; the lighter side gets a ground particle at cost R
    that absorbs the pT imbalance; then a balanced EMD is solved.
    """
    total_t, total_r = float(np.sum(t_pt)), float(np.sum(r_pt))
    excess = abs(total_t - total_r)
    if total_t >= total_r:
        r_eta, r_phi = np.append(r_eta, 0.0), np.append(r_phi, 0.0)
        r_w, t_w = np.append(r_pt, excess).astype(np.float64), np.asarray(t_pt, np.float64)
    else:
        t_eta, t_phi = np.append(t_eta, 0.0), np.append(t_phi, 0.0)
        t_w, r_w = np.append(t_pt, excess).astype(np.float64), np.asarray(r_pt, np.float64)
    total = t_w.sum()
    if total <= 0:
        return 0.0
    dr = np.hypot(t_eta[:, None] - r_eta[None, :], delta_phi(t_phi[:, None], r_phi[None, :]))
    cost = np.minimum(dr, R).astype(np.float64)
    if total_t >= total_r:
        cost[:, -1] = R
    else:
        cost[-1, :] = R
    return float(ot.emd2(t_w / total, r_w / total, cost) * total)


def emd(truth_objs: ak.Array, reco_objs: ak.Array, R: float = 1.0) -> np.ndarray:
    """`emd_event` for every event."""
    parts = []
    for objs in (truth_objs, reco_objs):
        c = counts(objs.pt)
        parts.append((offsets(c), flat(objs.pt), flat(objs.eta), flat(objs.phi)))
    (to, tpt, teta, tphi), (ro, rpt, reta, rphi) = parts
    out = np.empty(len(to) - 1)
    for i in range(len(out)):
        a, b, c, d = to[i], to[i + 1], ro[i], ro[i + 1]
        out[i] = emd_event(tpt[a:b], teta[a:b], tphi[a:b], rpt[c:d], reta[c:d], rphi[c:d], R)
    return out
