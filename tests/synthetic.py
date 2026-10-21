"""Synthetic truth and predictions with known answers, written as contract-v1 files."""

from __future__ import annotations

from pathlib import Path

import awkward as ak
import numpy as np

from scimo_bench.io import write_predictions
from scimo_bench.schema import Meta

FIRST_ID = 1_000_000


def make_truth(n_events: int = 500, seed: int = 0, mean_n: float = 7.0) -> ak.Array:
    """Jets of 1 + Poisson(mean_n - 1) particles around a random axis; pT ≥ 1 GeV, |η| < 2.4.

    Particles of one event are ≥ 0.15 apart in ΔR, so matching is unambiguous.
    """
    rng = np.random.default_rng(seed)
    pts, etas, phis = [], [], []
    for _ in range(n_events):
        n = 1 + rng.poisson(mean_n - 1)
        axis_eta, axis_phi = rng.uniform(-1.5, 1.5), rng.uniform(-np.pi, np.pi)
        eta, phi = [], []
        while len(eta) < n:
            e, p = axis_eta + rng.normal(0, 0.4), axis_phi + rng.normal(0, 0.4)
            if abs(e) >= 2.4:
                continue
            if all(
                np.hypot(e - a, np.angle(np.exp(1j * (p - b)))) > 0.15
                for a, b in zip(eta, phi, strict=True)
            ):
                eta.append(e)
                phi.append(p)
        pts.append(1.0 + rng.exponential(5.0, n))
        etas.append(np.array(eta))
        phis.append(np.angle(np.exp(1j * np.array(phi))))
    c = np.array([len(p) for p in pts])
    pt, eta, phi = (np.concatenate(x) for x in (pts, etas, phis))
    particles = ak.zip(
        {
            "pt": ak.unflatten(pt.astype(np.float32), c),
            "eta": ak.unflatten(eta.astype(np.float32), c),
            "phi": ak.unflatten(phi.astype(np.float32), c),
            "e": ak.unflatten((pt * np.cosh(eta)).astype(np.float32), c),
        }
    )
    ids = np.arange(FIRST_ID, FIRST_ID + n_events, dtype=np.int64)
    return ak.zip({"event_id": ids, "particles": particles}, depth_limit=1)


def as_prediction(truth: ak.Array, particles: ak.Array | None = None) -> ak.Array:
    return ak.zip(
        {
            "event_id": truth.event_id,
            "sample_id": ak.Array(np.zeros(len(truth), dtype=np.int32)),
            "particles": truth.particles if particles is None else particles,
        },
        depth_limit=1,
    )


def _map(truth, fn) -> ak.Array:
    """Apply fn(dict of flat numpy fields) -> dict to the particles."""
    p = truth.particles
    c = ak.to_numpy(ak.num(p, axis=1))
    flat = {k: ak.to_numpy(ak.flatten(p[k])).astype(np.float64) for k in p.fields}
    out = fn(flat)
    return ak.zip({k: ak.unflatten(np.asarray(v, np.float32), c) for k, v in out.items()})


def identical(truth) -> ak.Array:
    return as_prediction(truth)


def shifted(truth, factor: float = 1.1) -> ak.Array:
    """pT and E scaled by `factor`."""
    return as_prediction(
        truth, _map(truth, lambda f: {**f, "pt": f["pt"] * factor, "e": f["e"] * factor})
    )


def smeared(truth, sigma: float = 0.1, seed: int = 1) -> ak.Array:
    """pT and E × (1 + σ·N(0,1)); angles untouched."""
    rng = np.random.default_rng(seed)

    def fn(f):
        s = 1 + sigma * rng.standard_normal(len(f["pt"]))
        return {**f, "pt": f["pt"] * s, "e": f["e"] * s}

    return as_prediction(truth, _map(truth, fn))


def dropped(truth, p: float = 0.2, seed: int = 2) -> ak.Array:
    """Each particle removed with probability p."""
    rng = np.random.default_rng(seed)
    keep = rng.random(int(ak.sum(ak.num(truth.particles)))) >= p
    mask = ak.unflatten(keep, ak.num(truth.particles))
    return as_prediction(truth, truth.particles[mask])


def missing_events(pred: ak.Array, n: int = 10, seed: int = 3) -> tuple[ak.Array, np.ndarray]:
    """Drop n random events; returns (prediction, dropped event ids)."""
    rng = np.random.default_rng(seed)
    drop = rng.choice(len(pred), n, replace=False)
    keep = np.ones(len(pred), dtype=bool)
    keep[drop] = False
    return pred[keep], ak.to_numpy(pred.event_id[drop])


def missing_field(pred: ak.Array, field: str = "e") -> ak.Array:
    p = pred.particles
    kept = ak.zip({k: p[k] for k in p.fields if k != field})
    return ak.with_field(pred, kept, "particles")


def meta(name: str, generation: dict | None = None, family: str = "regression") -> Meta:
    return Meta(
        model=dict(name=name, family=family),
        generation=generation or dict(mode="synthetic", multiplicity="model"),
    )


def write(directory: Path, name: str, arr: ak.Array, generation: dict | None = None) -> Path:
    """Write a contract directory `directory/name`; returns the directory."""
    d = Path(directory) / name
    write_predictions(d, arr, meta(name, generation))
    return d


def write_truth(directory: Path, truth: ak.Array) -> Path:
    d = Path(directory) / "truth"
    write_predictions(d, truth, Meta(model=dict(name="truth")))
    return d
