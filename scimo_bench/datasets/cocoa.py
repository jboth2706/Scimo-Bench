"""COCOA single-jet dataset: truth reader for the raw ROOT files and its variables.

Truth: tree `EventTree`, one entry per event, particles as `particle_*` branches (GeV),
event id `eventNumber`. Collection `particles` with `pt, eta, phi, e, pdgid`.

Variables: the jet (vector sum of all particles, one per event), per-particle quantities,
greedy ΔR matching (task parameter `match_dr`) and the event EMD (`emd_r`). Add a variable
with `@REGISTRY.variable(...)` below; it can then be scored by a task or plotted by name.
"""

from __future__ import annotations

import os

import awkward as ak
import numpy as np
import uproot

from scimo_bench import io
from scimo_bench.schema import EVENT_ID
from scimo_bench.variables import (
    GENERIC,
    Context,
    Registry,
    counts,
    delta_phi,
    emd_event,
    f64,
    flat,
    match,
    obj_field,
    offsets,
    ratio,
    unflatten,
)

BRANCHES = {
    "pt": "particle_pt",
    "eta": "particle_eta",
    "phi": "particle_phi",
    "e": "particle_e",
    "pdgid": "particle_pdgid",
}
EVENT_BRANCH = "eventNumber"

REGISTRY = Registry("cocoa", parent=GENERIC)


class CocoaTruth:
    """Reads COCOA raw ROOT truth → `event_id` + `particles.{pt, eta, phi, e, pdgid}`."""

    def __init__(self, path: str | os.PathLike, tree: str = "EventTree"):
        self.path = path
        self.tree = tree

    def read(self, event_ids: np.ndarray | None = None) -> ak.Array:
        """The file's events (only `event_ids` if given), in file order."""
        with uproot.open(io.resolve_path(self.path)) as f:
            raw = f[self.tree].arrays([EVENT_BRANCH, *BRANCHES.values()])
        if event_ids is not None:
            raw = raw[np.isin(ak.to_numpy(raw[EVENT_BRANCH]), np.asarray(event_ids))]
        particles = ak.zip({k: raw[b] for k, b in BRANCHES.items()})
        ids = ak.values_astype(raw[EVENT_BRANCH], np.int64)
        return ak.zip({EVENT_ID: ids, "particles": particles}, depth_limit=1)


# --- helpers -------------------------------------------------------------------------------


def jet(arr: ak.Array, ctx: Context, side: str) -> dict[str, np.ndarray]:
    """The event's jet: massless vector sum of all objects (as `pflow_report._compute_jets`);
    `pt`, `eta`, `phi` per event. Cached per side ("truth" / "pred")."""
    key = ("jet", side)
    if key not in ctx.cache:
        pt, eta, phi = (f64(obj_field(arr, k, ctx)) for k in ("pt", "eta", "phi"))
        px = ak.to_numpy(ak.sum(pt * np.cos(phi), axis=1))
        py = ak.to_numpy(ak.sum(pt * np.sin(phi), axis=1))
        pz = ak.to_numpy(ak.sum(pt * np.sinh(eta), axis=1))
        jpt = np.hypot(px, py)
        safe = np.where(jpt == 0, 1e-10, jpt)
        ctx.cache[key] = dict(pt=jpt, eta=np.arcsinh(pz / safe), phi=np.arctan2(py, px))
    return ctx.cache[key]


def jet_energy(arr: ak.Array, ctx: Context) -> np.ndarray:
    """Sum of the objects' energies per event."""
    return ak.to_numpy(ak.sum(f64(obj_field(arr, "e", ctx)), axis=1))


def matched(truth: ak.Array, pred: ak.Array, ctx: Context) -> dict[str, np.ndarray]:
    """Greedy ΔR matching with the task's `match_dr`."""
    return match(truth, pred, ctx, ctx.param("match_dr"))


# --- jets ----------------------------------------------------------------------------------


@REGISTRY.variable("jet_truth_pt", inputs="truth", label=r"truth jet $p_T$", unit="GeV")
def _jet_truth_pt(truth, pred, ctx):
    """pT of the truth jet."""
    return jet(truth, ctx, "truth")["pt"]


@REGISTRY.variable("jet_truth_eta", inputs="truth", label=r"truth jet $\eta$")
def _jet_truth_eta(truth, pred, ctx):
    """η of the truth jet."""
    return jet(truth, ctx, "truth")["eta"]


@REGISTRY.variable("jet_truth_phi", inputs="truth", label=r"truth jet $\phi$")
def _jet_truth_phi(truth, pred, ctx):
    """φ of the truth jet."""
    return jet(truth, ctx, "truth")["phi"]


@REGISTRY.variable("jet_truth_e", inputs="truth", label="truth jet $E$", unit="GeV")
def _jet_truth_e(truth, pred, ctx):
    """Energy of the truth jet."""
    return jet_energy(truth, ctx)


@REGISTRY.variable("jet_pred_pt", label=r"jet $p_T$", unit="GeV", truth="jet_truth_pt")
def _jet_pred_pt(truth, pred, ctx):
    """pT of the predicted jet."""
    return jet(pred, ctx, "pred")["pt"]


@REGISTRY.variable("jet_pred_eta", label=r"jet $\eta$", truth="jet_truth_eta")
def _jet_pred_eta(truth, pred, ctx):
    """η of the predicted jet."""
    return jet(pred, ctx, "pred")["eta"]


@REGISTRY.variable("jet_pred_phi", label=r"jet $\phi$", truth="jet_truth_phi")
def _jet_pred_phi(truth, pred, ctx):
    """φ of the predicted jet."""
    return jet(pred, ctx, "pred")["phi"]


@REGISTRY.variable("jet_pred_e", label="jet $E$", unit="GeV", truth="jet_truth_e")
def _jet_pred_e(truth, pred, ctx):
    """Energy of the predicted jet."""
    return jet_energy(pred, ctx)


@REGISTRY.variable(
    "jet_energy_response", label=r"$E_\mathrm{pred} / E_\mathrm{truth}$ (jet)", reference=1.0
)
def _jet_energy_response(truth, pred, ctx):
    """Predicted over true jet energy."""
    return ratio(jet_energy(pred, ctx), jet_energy(truth, ctx))


@REGISTRY.variable(
    "jet_pt_response", label=r"$p_{T,\mathrm{pred}} / p_{T,\mathrm{truth}}$ (jet)", reference=1.0
)
def _jet_pt_response(truth, pred, ctx):
    """Predicted over true jet pT."""
    return ratio(jet(pred, ctx, "pred")["pt"], jet(truth, ctx, "truth")["pt"])


@REGISTRY.variable("jet_dr", label=r"$\Delta R$(pred jet, truth jet)", reference=0.0)
def _jet_dr(truth, pred, ctx):
    """ΔR between predicted and true jet axis."""
    t, p = jet(truth, ctx, "truth"), jet(pred, ctx, "pred")
    return np.hypot(p["eta"] - t["eta"], delta_phi(p["phi"], t["phi"]))


# --- particles -----------------------------------------------------------------------------


@REGISTRY.variable(
    "particle_truth_pt", level="object", inputs="truth", label=r"truth particle $p_T$", unit="GeV"
)
def _particle_truth_pt(truth, pred, ctx):
    """pT of each truth particle."""
    return obj_field(truth, "pt", ctx)


@REGISTRY.variable(
    "particle_truth_eta", level="object", inputs="truth", label=r"truth particle $\eta$"
)
def _particle_truth_eta(truth, pred, ctx):
    """η of each truth particle."""
    return obj_field(truth, "eta", ctx)


@REGISTRY.variable(
    "particle_pred_pt",
    level="object",
    label=r"particle $p_T$",
    unit="GeV",
    truth="particle_truth_pt",
)
def _particle_pred_pt(truth, pred, ctx):
    """pT of each predicted particle."""
    return obj_field(pred, "pt", ctx)


@REGISTRY.variable(
    "particle_pred_eta", level="object", label=r"particle $\eta$", truth="particle_truth_eta"
)
def _particle_pred_eta(truth, pred, ctx):
    """η of each predicted particle."""
    return obj_field(pred, "eta", ctx)


@REGISTRY.variable(
    "particle_matched", level="object", label="truth particle matched", reference=1.0
)
def _particle_matched(truth, pred, ctx):
    """1 per truth particle with a matched prediction, else 0 (→ efficiency)."""
    m = matched(truth, pred, ctx)
    return unflatten((m["truth_idx"] >= 0).astype(np.float64), m["truth_counts"])


@REGISTRY.variable(
    "particle_fake", level="object", label="predicted particle unmatched", reference=0.0
)
def _particle_fake(truth, pred, ctx):
    """1 per predicted particle without a truth match, else 0 (→ fake rate)."""
    m = matched(truth, pred, ctx)
    return unflatten((~m["pred_matched"]).astype(np.float64), m["pred_counts"])


@REGISTRY.variable(
    "particle_dr", level="object", label=r"$\Delta R$(matched particles)", reference=0.0
)
def _particle_dr(truth, pred, ctx):
    """ΔR to the matched prediction, per truth particle (NaN if unmatched)."""
    m = matched(truth, pred, ctx)
    return unflatten(m["truth_dr"], m["truth_counts"])


@REGISTRY.variable(
    "particle_pt_response",
    level="object",
    label=r"$p_{T,\mathrm{pred}} / p_{T,\mathrm{truth}}$ (matched particles)",
    reference=1.0,
)
def _particle_pt_response(truth, pred, ctx):
    """Matched predicted over true pT, per truth particle (NaN if unmatched)."""
    m = matched(truth, pred, ctx)
    idx = m["truth_idx"]
    p_pt = flat(obj_field(pred, "pt", ctx))
    t_pt = flat(obj_field(truth, "pt", ctx))
    resp = np.full(len(idx), np.nan)
    ok = idx >= 0
    resp[ok] = p_pt[idx[ok]] / t_pt[ok]
    return unflatten(resp, m["truth_counts"])


# --- event distance ------------------------------------------------------------------------


@REGISTRY.variable("event_emd", label="EMD (truth, pred)", unit="GeV", reference=0.0)
def _event_emd(truth, pred, ctx):
    """Unbalanced EMD between true and predicted particles, per event (`emd_r` caps ΔR)."""
    parts = []
    for arr in (truth, pred):
        vals = [obj_field(arr, k, ctx) for k in ("pt", "eta", "phi")]
        c = counts(vals[0])
        parts.append((offsets(c), *(flat(v) for v in vals)))
    (to, tpt, teta, tphi), (po, ppt, peta, pphi) = parts
    r = ctx.param("emd_r")
    out = np.empty(len(to) - 1)
    for i in range(len(out)):
        a, b, c, d = to[i], to[i + 1], po[i], po[i + 1]
        out[i] = emd_event(tpt[a:b], teta[a:b], tphi[a:b], ppt[c:d], peta[c:d], pphi[c:d], r)
    return out
