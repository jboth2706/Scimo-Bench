"""Physics helpers shared by datasets: jets, ΔR and array conversions.

Inputs are awkward arrays of objects (jagged: a list per event) with fields such as `pt`,
`eta`, `phi`; outputs are numpy arrays (one value per event) or jagged arrays.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

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
