"""Variables: every physics quantity, as named functions in a registry.

A variable receives `(truth, pred, ctx)` — truth and prediction rows aligned one to one,
after cuts — and returns one value per event (numpy) or a jagged array per object
(awkward). A variable that needs a field the input lacks raises `MissingInput`.
Collection variables (`inputs="objects"`) receive `(objects, ctx)`; cuts are built on them.

Each dataset has its own registry (e.g. `scimo_bench/datasets/cocoa.py`) whose parent is
`GENERIC` (the variables any collection with `pt`, `eta` has). The run activates its
dataset's registry with `use()`; `get`, `exists`, `info` look names up there. To add a
variable, write a function in the dataset's module:

    @REGISTRY.variable("jet_mass", label="jet mass", unit="GeV")
    def jet_mass(truth, pred, ctx):
        return ...  # one value per event
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import awkward as ak
import numpy as np
import ot

LEVELS = ("event", "object")
INPUTS = ("pair", "truth", "objects")


class MissingInput(Exception):
    """A variable needs an input field that is not there."""

    def __init__(self, field_name: str):
        super().__init__(f"missing input '{field_name}'")
        self.field = field_name


@dataclass
class Context:
    """What variables of one (model, selection) share: the collection they work on, the
    task's parameters (e.g. `match_dr`) and a cache for intermediate results."""

    collection: str = "particles"
    params: dict = field(default_factory=dict)
    cache: dict = field(default_factory=dict)

    @classmethod
    def from_params(cls, params: dict) -> Context:
        """From a task's `params` block: `collection` plus any other parameters."""
        params = dict(params)
        return cls(collection=params.pop("collection", "particles"), params=params)

    def param(self, name: str):
        """A task parameter; a variable that needs a missing one is a configuration error."""
        if name not in self.params:
            raise KeyError(f"task parameter '{name}' is not set (task.params.{name})")
        return self.params[name]


@dataclass(frozen=True)
class Variable:
    """A registered variable: its function plus metadata for scoring and plotting."""

    name: str
    fn: Callable
    level: str  # "event" (one value per event) or "object" (one per object)
    inputs: str  # "pair" (truth + pred), "truth" (truth only), "objects" (one collection)
    label: str
    unit: str
    reference: float | None  # the ideal value (e.g. 1 for a response), drawn as a line
    truth: str | None  # the truth counterpart of a prediction variable (for overlays)

    def __call__(self, *args, ctx: Context):
        """Evaluate, cached per context (collection variables are not cached)."""
        key = ("var", self.name)
        if self.inputs != "objects" and key in ctx.cache:
            return ctx.cache[key]
        out = self.fn(*args, ctx)
        if self.inputs != "objects":
            ctx.cache[key] = out
        return out

    @property
    def needs_pred(self) -> bool:
        """Whether the variable needs predictions (else truth only)."""
        return self.inputs == "pair"

    @property
    def axis_label(self) -> str:
        """Label with unit, for axes."""
        return f"{self.label} [{self.unit}]" if self.unit else self.label


@dataclass(frozen=True)
class VariableInfo:
    """Metadata of a variable, without its function (what the plotter may use)."""

    name: str
    level: str
    label: str
    unit: str
    reference: float | None
    truth: str | None
    axis_label: str
    truth_only: bool


class Registry:
    """Named variables of one dataset; names not found here are looked up in `parent`."""

    def __init__(self, name: str, parent: Registry | None = None):
        self.name = name
        self.parent = parent
        self.vars: dict[str, Variable] = {}

    def variable(
        self,
        name: str,
        level: str = "event",
        inputs: str = "pair",
        label: str | None = None,
        unit: str = "",
        reference: float | None = None,
        truth: str | None = None,
    ):
        """Decorator registering a function `(truth, pred, ctx)` (or `(objects, ctx)` for
        `inputs="objects"`) as variable `name`."""
        assert level in LEVELS and inputs in INPUTS, (level, inputs)

        def wrap(fn):
            if name in self.vars:
                raise ValueError(f"variable '{name}' registered twice in {self.name}")
            self.vars[name] = Variable(
                name, fn, level, inputs, label or name, unit, reference, truth
            )
            return fn

        return wrap

    def names(self) -> list[str]:
        """All variable names visible from this registry (own and parents')."""
        own = sorted(self.vars)
        return own + [n for n in (self.parent.names() if self.parent else []) if n not in own]

    def exists(self, name: str) -> bool:
        return name in self.vars or (self.parent is not None and self.parent.exists(name))

    def get(self, name: str) -> Variable:
        if name in self.vars:
            return self.vars[name]
        if self.parent is not None and self.parent.exists(name):
            return self.parent.get(name)
        raise KeyError(f"unknown variable '{name}' for {self.name}; known: {self.names()}")


GENERIC = Registry("generic")
_active: Registry = GENERIC


def use(registry: Registry) -> None:
    """Make `registry` the one `get`, `exists` and `info` look names up in."""
    global _active
    _active = registry


def active() -> Registry:
    """The active registry."""
    return _active


def get(name: str) -> Variable:
    """The variable `name` of the active registry."""
    return _active.get(name)


def exists(name: str) -> bool:
    return _active.exists(name)


def info(name: str) -> VariableInfo:
    """Metadata of variable `name` (label, unit, reference, truth counterpart)."""
    v = get(name)
    return VariableInfo(
        v.name, v.level, v.label, v.unit, v.reference, v.truth, v.axis_label, v.inputs == "truth"
    )


# --- helpers for writing variables ---------------------------------------------------------


def objects(arr: ak.Array, ctx: Context) -> ak.Array:
    """The context's collection of `arr` (jagged objects per event)."""
    if ctx.collection not in arr.fields:
        raise MissingInput(ctx.collection)
    return arr[ctx.collection]


def obj_field(arr: ak.Array, name: str, ctx: Context) -> ak.Array:
    """One field of the context's collection, e.g. `pt` (jagged)."""
    objs = objects(arr, ctx)
    if name not in objs.fields:
        raise MissingInput(f"{ctx.collection}.{name}")
    return objs[name]


def coll_field(objs: ak.Array, name: str) -> ak.Array:
    """One field of a collection already selected (for collection variables)."""
    if name not in objs.fields:
        raise MissingInput(name)
    return objs[name]


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


def delta_phi(a, b):
    """φ difference wrapped to (−π, π]."""
    d = a - b
    return np.arctan2(np.sin(d), np.cos(d))


def ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num / den, inf/NaN where den is 0 (dropped later as non-finite)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den


def match(truth: ak.Array, pred: ak.Array, ctx: Context, max_dr: float) -> dict[str, np.ndarray]:
    """Greedy ΔR matching: truth objects by descending pT take the nearest free prediction
    within `max_dr`. Flat arrays: `truth_idx` (matched prediction, flat index or -1),
    `truth_dr` (NaN if unmatched), `pred_matched`; plus both count arrays. Cached."""
    if ("match",) in ctx.cache:
        return ctx.cache[("match",)]
    t_pt, t_eta, t_phi = (obj_field(truth, k, ctx) for k in ("pt", "eta", "phi"))
    p_eta, p_phi = obj_field(pred, "eta", ctx), obj_field(pred, "phi", ctx)
    tc, pc = counts(t_pt), counts(p_eta)
    to, po = offsets(tc), offsets(pc)
    tpt, teta, tphi = flat(t_pt), flat(t_eta), flat(t_phi)
    peta, pphi = flat(p_eta), flat(p_phi)
    truth_idx = np.full(len(tpt), -1, dtype=np.int64)
    truth_dr = np.full(len(tpt), np.nan)
    pred_matched = np.zeros(len(peta), dtype=bool)
    for i in np.flatnonzero((tc > 0) & (pc > 0)):
        a, b, c, d = to[i], to[i + 1], po[i], po[i + 1]
        dr = np.hypot(
            teta[a:b, None] - peta[None, c:d], delta_phi(tphi[a:b, None], pphi[None, c:d])
        )
        for k in np.argsort(-tpt[a:b], kind="stable"):
            row = np.where(pred_matched[c:d], np.inf, dr[k])
            j = int(np.argmin(row))
            if row[j] < max_dr:
                pred_matched[c + j] = True
                truth_idx[a + k] = c + j
                truth_dr[a + k] = row[j]
    out = dict(
        truth_idx=truth_idx,
        truth_dr=truth_dr,
        pred_matched=pred_matched,
        truth_counts=tc,
        pred_counts=pc,
    )
    ctx.cache[("match",)] = out
    return out


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


# --- generic variables (any collection with pt, eta) ---------------------------------------


@GENERIC.variable("pt", level="object", inputs="objects", label="$p_T$", unit="GeV")
def _pt(objs, ctx):
    """Object pT (for cuts)."""
    return coll_field(objs, "pt")


@GENERIC.variable("abs_eta", level="object", inputs="objects", label=r"$|\eta|$")
def _abs_eta(objs, ctx):
    """Object |η| (for cuts)."""
    return abs(coll_field(objs, "eta"))


@GENERIC.variable("n_objects", inputs="objects", label="objects per event")
def _n_objects(objs, ctx):
    """Objects per event (for event-level cuts)."""
    return counts(objs)


@GENERIC.variable("n_truth", inputs="truth", label="truth objects per event")
def _n_truth(truth, pred, ctx):
    """Truth multiplicity."""
    return counts(objects(truth, ctx)).astype(np.float64)


@GENERIC.variable("n_pred", label="objects per event", truth="n_truth")
def _n_pred(truth, pred, ctx):
    """Predicted multiplicity."""
    return counts(objects(pred, ctx)).astype(np.float64)


@GENERIC.variable("multiplicity_diff", label=r"$N_\mathrm{pred} - N_\mathrm{truth}$", reference=0.0)
def _multiplicity_diff(truth, pred, ctx):
    """Predicted minus true multiplicity."""
    return get("n_pred")(truth, pred, ctx=ctx) - get("n_truth")(truth, pred, ctx=ctx)
