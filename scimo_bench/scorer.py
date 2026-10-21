"""Metrics with paired-bootstrap intervals, status, rankings and the report tables.

Every statistic is computed with integer event weights: weight 1 for every event gives the
point estimate (equal to the numpy reference), multinomial counts from one resample of the
events give one bootstrap replica. All models share the same resamples (paired bootstrap).
The scorer computes no physics; it calls the dataset's observables and comparisons
(`scimo_bench/catalogue.py`).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

import awkward as ak
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from scimo_bench.catalogue import Catalogue, Missing, call_comparison, missing_field
from scimo_bench.io import Joined

log = logging.getLogger(__name__)

REPORT_COLUMNS = [
    "model",
    "metric",
    "bin_low",
    "bin_high",
    "value",
    "lower",
    "upper",
    "status",
    "kind",
    "task_version",
    "condition",
]
RANKING_COLUMNS = ["metric", "model", "value", "status", "rank", "primary", "normalized"]
OK, MISSING, INVALID = "ok", "missing", "invalid"
TRUTH = "truth"


# --- weighted statistics -------------------------------------------------------------------


@dataclass
class Sample:
    """Finite values of one (model, metric, bin), sorted, with the event of each value."""

    values: np.ndarray
    events: np.ndarray

    @classmethod
    def build(cls, values: np.ndarray, events: np.ndarray) -> Sample:
        order = np.argsort(values, kind="stable")
        return cls(values[order], events[order])

    def __len__(self) -> int:
        return len(self.values)


def weighted_quantile(s: Sample, w: np.ndarray, q: float) -> float:
    """Quantile (numpy `linear`) of the sample with each value repeated by its weight."""
    cw = np.cumsum(w)
    n = cw[-1] if len(cw) else 0
    if n == 0:
        return np.nan
    h = (n - 1) * q
    k0 = np.floor(h)
    k1 = min(k0 + 1, n - 1)
    v0 = s.values[np.searchsorted(cw, k0, side="right")]
    v1 = s.values[np.searchsorted(cw, k1, side="right")]
    return float(v0 + (h - k0) * (v1 - v0))


def weighted_mean(s: Sample, w: np.ndarray) -> float:
    """Mean with integer event weights (bootstrap counts)."""
    n = w.sum()
    return float(np.dot(w, s.values) / n) if n else np.nan


def weighted_std(s: Sample, w: np.ndarray) -> float:
    """Standard deviation with integer event weights."""
    n = w.sum()
    if not n:
        return np.nan
    mu = np.dot(w, s.values) / n
    return float(np.sqrt(np.dot(w, (s.values - mu) ** 2) / n))


# --- metrics -------------------------------------------------------------------------------


class Metric:
    """A statistic of one quantity, optionally in bins of `x` (explicit edges).

    `variable` and `x` name a comparison (`jet_e_response`) or an observable with its side
    (`truth.jet_pt`, `reco.particle_pt`)."""

    default_better: str | Mapping = "lower"
    location = False  # True: the value estimates the variable itself (its reference applies)

    def __init__(
        self,
        name: str,
        variable: str,
        better: str | Mapping | None = None,
        x: str | None = None,
        binning: list[float] | None = None,
        label: str | None = None,
    ):
        self.name = name
        self.variable = variable
        self.better = self.default_better if better is None else better
        self.x = x
        self.binning = None if binning is None else [float(b) for b in binning]
        self.label = label or name
        if (x is None) != (binning is None):
            raise ValueError(f"metric '{name}': give both x and binning, or neither")
        if self.binning is not None and (
            len(self.binning) < 2 or np.any(np.diff(self.binning) <= 0)
        ):
            raise ValueError(f"metric '{name}': binning must be increasing edges")
        better_key(self.better, np.array([0.0]))  # validates `better`

    def stat(self, s: Sample, w: np.ndarray) -> float:
        raise NotImplementedError

    def __call__(self, s: Sample, event_weights: np.ndarray) -> float:
        if len(s) == 0:
            return np.nan
        return self.stat(s, event_weights[s.events])

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, {self.variable!r})"


class Quantile(Metric):
    """A quantile of the variable (e.g. q=0.5)."""

    location = True

    def __init__(
        self,
        name: str,
        variable: str,
        q: float = 0.5,
        better: str | Mapping | None = None,
        x: str | None = None,
        binning: list[float] | None = None,
        label: str | None = None,
    ):
        super().__init__(name, variable, better, x, binning, label)
        self.q = float(q)

    def stat(self, s, w):
        return weighted_quantile(s, w, self.q)


class Median(Metric):
    """Median of the variable."""

    location = True

    def stat(self, s, w):
        return weighted_quantile(s, w, 0.5)


class IQR(Metric):
    """Interquartile range (q75 − q25) of the variable."""

    def stat(self, s, w):
        return weighted_quantile(s, w, 0.75) - weighted_quantile(s, w, 0.25)


class Mean(Metric):
    """Mean of the variable."""

    location = True

    def stat(self, s, w):
        return weighted_mean(s, w)


class Std(Metric):
    """Standard deviation of the variable."""

    def stat(self, s, w):
        return weighted_std(s, w)


class Efficiency(Mean):
    """Fraction of truth objects matched (mean of `particle_matched`)."""

    default_better = "higher"


class FakeRate(Mean):
    """Fraction of predicted objects without a truth match (mean of `particle_fake`)."""

    default_better = "lower"


def better_key(better, values: np.ndarray) -> np.ndarray:
    """Map values to a key where lower is better."""
    values = np.asarray(values, dtype=float)
    if isinstance(better, Mapping):
        if set(better) != {"closest_to"}:
            raise ValueError(f"better must be lower|higher|{{closest_to: v}}, got {better}")
        return np.abs(values - float(better["closest_to"]))
    if better == "lower":
        return values
    if better == "higher":
        return -values
    raise ValueError(f"better must be lower|higher|{{closest_to: v}}, got {better!r}")


# --- values per model ----------------------------------------------------------------------


@dataclass
class ModelInfo:
    """What the scorer knows about a configured model."""

    name: str
    condition: str
    available: bool = True
    reason: str = ""


def compute(fn, args: tuple, who: str, name: str, required: set[str]):
    """`fn(*args)`, or `Missing` if its input lacks a field: a warning if the field is one
    the dataset requires from models (`required`), else (e.g. tracks, an input) silent."""
    try:
        return fn(*args)
    except Exception as e:
        field_name = missing_field(e)
        if field_name is None:
            raise
        if field_name in required:
            log.warning("%s: %s is missing (no field '%s')", who, name, field_name)
        else:
            log.debug("%s: %s not computed (no field '%s')", who, name, field_name)
        return Missing(f"no field '{field_name}'")


def evaluate(
    joined: Joined, cat: Catalogue, params: Mapping, required: set[str] = frozenset()
) -> dict[str, dict]:
    """Every observable and comparison: {'truth': {observable: values}, model:
    {'truth.<obs>' (truth rows of the model), 'reco.<obs>', <comparison>: values}};
    a value that cannot be computed is a `Missing`."""
    truth_vals = {
        name: compute(fn, (joined.truth,), TRUTH, name, required)
        for name, fn in cat.observables.items()
    }
    out: dict[str, dict] = {TRUTH: truth_vals}
    for model in joined.models:
        rows = joined.rows[model]
        truth, reco = joined.truth_rows(model), joined.preds[model]
        vals: dict = {}
        for name, fn in cat.observables.items():
            tv = truth_vals[name]
            vals[f"truth.{name}"] = tv if isinstance(tv, Missing) else tv[rows]
            vals[f"reco.{name}"] = compute(fn, (reco,), model, name, required)
        for name, fn in cat.comparisons.items():
            args = (fn, truth, reco, params)
            vals[name] = compute(call_comparison, args, model, name, required)
        out[model] = vals
    return out


def flatten(values, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """(flat values, event index per value, counts per row or None for event level)."""
    if isinstance(values, ak.Array) and values.ndim > 1:
        c = ak.to_numpy(ak.num(values, axis=1))
        flat = ak.to_numpy(ak.flatten(values, axis=1)).astype(np.float64)
        return flat, np.repeat(rows, c), c
    return np.asarray(values, dtype=np.float64), rows, None


def samples_for(metric: Metric, vals: dict, rows: np.ndarray) -> list[Sample]:
    """Overall sample, then one per bin. Raises Missing / ValueError."""
    v = vals[metric.variable]
    if isinstance(v, Exception):
        raise v
    values, events, c = flatten(v, rows)
    if metric.x is None:
        ok = np.isfinite(values)
        return [Sample.build(values[ok], events[ok])]
    x = vals[metric.x]
    if isinstance(x, Exception):
        raise x
    xv, _, xc = flatten(x, rows)
    if len(xv) != len(values) or (c is None) != (xc is None) or (c is not None and np.any(c != xc)):
        raise ValueError(f"'{metric.variable}' and '{metric.x}' differ in structure")
    ok = np.isfinite(values) & np.isfinite(xv)
    edges = np.asarray(metric.binning)
    inside = ok & (xv >= edges[0]) & (xv < edges[-1])
    out = [Sample.build(values[ok], events[ok])]
    idx = np.searchsorted(edges, xv, side="right") - 1
    for b in range(len(edges) - 1):
        m = inside & (idx == b)
        out.append(Sample.build(values[m], events[m]))
    return out


# --- scoring -------------------------------------------------------------------------------


@dataclass
class ScoreResult:
    """Report rows, all values (`evaluate`) and the per-event table of one scoring."""

    report: pd.DataFrame
    values: dict[str, dict]
    per_event: ak.Array
    problems: list[str] = field(default_factory=list)


def score(
    joined: Joined,
    models: list[ModelInfo],
    metrics: list[Metric],
    cat: Catalogue,
    params: Mapping | None = None,
    required: set[str] = frozenset(),
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 0,
    task_version: int = 1,
) -> ScoreResult:
    """All metrics for all models, with paired bootstrap intervals; `params` are the task
    parameters for the comparisons, `required` the fields models must output."""
    vals = evaluate(joined, cat, params or {}, required)
    n_events = len(joined.event_ids)

    # (model, metric) -> list of samples (overall + bins), or a status and reason
    samples: dict[tuple[str, str], list[Sample]] = {}
    status: dict[tuple[str, str], str] = {}
    for info in models:
        for m in metrics:
            key = (info.name, m.name)
            if not info.available:
                status[key] = MISSING
                continue
            try:
                samples[key] = samples_for(m, vals[info.name], joined.rows[info.name])
                status[key] = OK
            except Missing:
                status[key] = MISSING
            except ValueError as e:
                log.warning("%s / %s: %s", info.name, m.name, e)
                status[key] = INVALID

    metric_by_name = {m.name: m for m in metrics}
    ones = np.ones(n_events)
    points = {
        key: np.array([metric_by_name[key[1]](s, ones) for s in ss]) for key, ss in samples.items()
    }

    boot = {key: np.full((n_boot, len(ss)), np.nan) for key, ss in samples.items()}
    if n_events and n_boot:
        rng = np.random.default_rng(seed)
        for b in range(n_boot):
            w = np.bincount(rng.integers(0, n_events, n_events), minlength=n_events).astype(float)
            for key, ss in samples.items():
                metric = metric_by_name[key[1]]
                boot[key][b] = [metric(s, w) for s in ss]

    alpha = 100 * (1 - level) / 2
    records = []
    for info in models:
        for m in metrics:
            key = (info.name, m.name)
            bins = [(np.nan, np.nan)]
            if m.binning is not None:
                bins += list(zip(m.binning[:-1], m.binning[1:], strict=True))
            for i, (lo, hi) in enumerate(bins):
                value = lower = upper = np.nan
                st = status[key]
                if st == OK:
                    value = points[key][i]
                    if not np.isfinite(value):
                        st = INVALID
                    else:
                        col = boot[key][:, i]
                        if np.isfinite(col).any():
                            lower, upper = np.nanpercentile(col, [alpha, 100 - alpha])
                records.append(
                    dict(
                        model=info.name,
                        metric=m.name,
                        bin_low=lo,
                        bin_high=hi,
                        value=value,
                        lower=lower,
                        upper=upper,
                        status=st,
                        kind="quality",
                        task_version=task_version,
                        condition=f"gen={info.condition}",
                    )
                )
    report = pd.DataFrame(records, columns=REPORT_COLUMNS)
    return ScoreResult(report, vals, per_event_array(joined, vals, cat))


def overall(report: pd.DataFrame) -> pd.DataFrame:
    """Unbinned rows only."""
    return report[report["bin_low"].isna() & report["bin_high"].isna()]


# --- rankings ------------------------------------------------------------------------------


def rank(
    report: pd.DataFrame,
    metrics: list[Metric],
    primary: str | None = None,
    normalization: Mapping | None = None,
) -> pd.DataFrame:
    """Rank models per metric (overall values) and by mean rank; missing/invalid last."""
    q = overall(report[report["kind"] == "quality"])
    models = list(dict.fromkeys(q["model"]))
    base = ref = None
    if normalization:
        base, ref = normalization.get("baseline"), normalization.get("reference")
        if base and ref and not {base, ref} <= set(models):
            log.warning("normalization: baseline/reference not among the models, skipped")
            base = ref = None
    records = []
    ranks: dict[str, list[float]] = {m: [] for m in models}
    for metric in metrics:
        rows = q[q["metric"] == metric.name].set_index("model").reindex(models)
        ok = (rows["status"] == OK).to_numpy()
        r = np.full(len(models), float(len(models)))
        if ok.any():
            r[ok] = rankdata(better_key(metric.better, rows["value"].to_numpy()[ok]), method="min")
        norm = np.full(len(models), np.nan)
        if base and ref:
            vb, vr = rows.loc[base, "value"], rows.loc[ref, "value"]
            if rows.loc[base, "status"] == OK and rows.loc[ref, "status"] == OK and vr != vb:
                norm = (rows["value"].to_numpy() - vb) / (vr - vb)
        for i, model in enumerate(models):
            ranks[model].append(r[i])
            records.append(
                dict(
                    metric=metric.name,
                    model=model,
                    value=rows["value"].iloc[i],
                    status=rows["status"].iloc[i],
                    rank=int(r[i]),
                    primary=metric.name == primary,
                    normalized=norm[i],
                )
            )
    mean = np.array([np.mean(ranks[m]) if ranks[m] else np.nan for m in models])
    overall_rank = rankdata(mean, method="min") if len(models) else []
    for i, model in enumerate(models):
        statuses = {r["status"] for r in records if r["model"] == model}
        records.append(
            dict(
                metric="mean_rank",
                model=model,
                value=mean[i],
                status=OK if statuses == {OK} else MISSING if MISSING in statuses else INVALID,
                rank=int(overall_rank[i]),
                primary=False,
                normalized=np.nan,
            )
        )
    return pd.DataFrame(records, columns=RANKING_COLUMNS)


# --- per-event table -----------------------------------------------------------------------


def _column(value, n: int):
    if isinstance(value, Exception) or value is None:
        return ak.Array([None] * n)
    return value if isinstance(value, ak.Array) else ak.Array(np.asarray(value))


def per_event_array(joined: Joined, vals: dict[str, dict], cat: Catalogue) -> ak.Array:
    """One record per (model, row) — and per truth event under model 'truth' — with every
    observable (of that side) and comparison (models only) as a column."""
    parts = []
    n = len(joined.event_ids)
    blocks = [(TRUTH, joined.event_ids, np.zeros(n, dtype=np.int64), n)]
    for model in joined.models:
        p = joined.preds[model]
        blocks.append((model, ak.to_numpy(p["event_id"]), ak.to_numpy(p["sample_id"]), len(p)))
    for model, ids, samples, rows in blocks:
        cols = {
            "model": ak.Array([model] * rows),
            "event_id": ak.Array(ids),
            "sample_id": ak.Array(samples),
        }
        for name in cat.observables:
            key = name if model == TRUTH else f"reco.{name}"
            cols[name] = _column(vals[model].get(key), rows)
        for name in cat.comparisons:
            cols[name] = _column(vals[model].get(name), rows)
        parts.append(ak.zip(cols, depth_limit=1))
    return ak.concatenate(parts) if parts else ak.Array([])
