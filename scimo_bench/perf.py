"""Inference cost of models: runtime, scenarios, timing rules, quality gate.

Models run through the model protocol (`protocol.py`). Timing rules: first call (compile)
and warmup are separate rows; CUDA is synchronized around every timed region; model time
(inputs already on device) and end-to-end time (inputs + generate + to_floats) are separate
metrics; every number has repeats and an interval. Rows are written ungated to `perf.csv`;
the quality gate is applied when they are merged into `report.csv` (`apply_gate`), so perf
may run before scoring.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from scimo_bench import protocol
from scimo_bench.schema import condition_hash
from scimo_bench.scorer import INVALID, MISSING, OK, REPORT_COLUMNS, overall

try:
    import torch
except ImportError:  # the default (quality) environment has no torch
    torch = None

log = logging.getLogger(__name__)

SCENARIOS = ("offline", "single")
STATIC_INFO = ("n_params", "flops_per_event", "nfe_per_event", "tokens_per_event")
MAX_SCATTER = 0.05
METRICS = {
    "offline": ("throughput", "throughput_e2e", "gpu_s_per_1k", "peak_memory"),
    "single": ("latency_p50", "latency_p90", "latency_p99", "latency_e2e_p50", "latency_e2e_p99"),
}
COMMON = ("load_s", "compile_s", "warmup_s", *STATIC_INFO)


@dataclass
class Runtime:
    """How model code runs (config group `runtime`)."""

    name: str
    device: str = "cuda"
    precision: str = "bf16"
    compile: bool = True
    batch_size: int = 256

    @classmethod
    def from_cfg(cls, cfg: Mapping) -> Runtime:
        r = cls(**{k: cfg[k] for k in cls.__dataclass_fields__ if k in cfg})
        if is_cuda(r.device) and (torch is None or not torch.cuda.is_available()):
            raise RuntimeError(
                f"runtime={r.name} needs torch with CUDA (a GPU job in an env with torch); "
                "use runtime=cpu otherwise"
            )
        return r

    def for_model(self) -> dict:
        """What the model's `load` gets as `runtime`."""
        return dict(device=self.device, precision=self.precision, compile=self.compile)


@dataclass
class PerfSettings:
    """What to measure (config group `perf`)."""

    name: str
    scenarios: list[str]
    batch_sizes: list[int]
    warmup: int = 20
    repeats: int = 5
    n_events: int = 20000
    single_events: int = 1000
    n_gpus: int = 1

    @classmethod
    def from_cfg(cls, cfg: Mapping) -> PerfSettings:
        s = cls(**{k: cfg[k] for k in cls.__dataclass_fields__ if k in cfg})
        s.scenarios, s.batch_sizes = list(s.scenarios), [int(b) for b in s.batch_sizes]
        bad = [x for x in s.scenarios if x not in SCENARIOS]
        if bad:
            raise ValueError(f"unknown perf scenarios {bad}; known: {SCENARIOS}")
        return s


def is_cuda(device: str) -> bool:
    """Whether a device string names a CUDA device."""
    return device.startswith("cuda")


def sync(device: str) -> None:
    """Wait for the device (CUDA) to finish."""
    if is_cuda(device):
        torch.cuda.synchronize()


def timed(fn, device: str, *args):
    """(result, seconds) with the device synchronized before and after."""
    sync(device)
    t0 = time.perf_counter()
    out = fn(*args)
    sync(device)
    return out, time.perf_counter() - t0


def hardware(rt: Runtime) -> dict:
    """Hardware fingerprint for `meta` and perf conditions."""
    hw: dict[str, Any] = dict(
        device=rt.device,
        precision=rt.precision,
        compiled=rt.compile,
        host=platform.node(),
        cpu=platform.processor() or platform.machine(),
        cpu_count=os.cpu_count(),
        torch=None if torch is None else torch.__version__,
    )
    if is_cuda(rt.device):
        hw.update(
            device_name=torch.cuda.get_device_name(0),
            gpu_count=torch.cuda.device_count(),
            cuda=torch.version.cuda,
            driver=_driver_version(),
        )
    else:
        hw.update(device_name=hw["cpu"], gpu_count=0)
    return hw


def _driver_version() -> str | None:
    """NVIDIA driver version, if nvidia-smi is there."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 else None
    except (OSError, IndexError, subprocess.TimeoutExpired):
        return None


def perf_condition(gen: str, scenario: str, batch_size: int, hw: dict) -> str:
    """The `condition` string of a perf row."""
    return (
        f"gen={gen};scenario={scenario};batch_size={batch_size};device={hw['device_name']};"
        f"precision={hw['precision']};compiled={hw['compiled']}"
    )


def chunks(events: np.ndarray, size: int) -> list[np.ndarray]:
    """Full batches of event ids (the last partial batch is dropped)."""
    return [events[i : i + size] for i in range(0, len(events) - size + 1, size)]


def summary(values: list[float]) -> tuple[float, float, float, float]:
    """(mean, min, max, relative scatter) over repeats."""
    v = np.asarray(values, dtype=float)
    mean = float(v.mean())
    scatter = float(v.std() / mean) if mean and len(v) > 1 else 0.0
    return mean, float(v.min()), float(v.max()), scatter


def load_model(model_cfg: Mapping, rt: Runtime) -> tuple[protocol.Model, float, dict]:
    """Create and load a protocol model: (model, load seconds, info())."""
    model = protocol.create(model_cfg["model"])
    _, load_s = timed(model.load, rt.device, {**dict(model_cfg), "runtime": rt.for_model()})
    return model, load_s, dict(model.info())


def measure(model, events: np.ndarray, scenario: str, batch_size: int, s: PerfSettings, device):
    """{metric: per-repeat values} plus warmup seconds for one scenario and batch size."""
    n = s.single_events if scenario == "single" else s.n_events
    batches_ids = chunks(events[:n], batch_size)
    if not batches_ids:
        raise ValueError(f"not enough events ({len(events)}) for batch size {batch_size}")
    batches = [model.inputs(ids, batch_size) for ids in batches_ids]
    t_warm = 0.0
    for i in range(s.warmup):
        _, dt = timed(model.generate, device, batches[i % len(batches)])
        t_warm += dt
    res: dict[str, list[float]] = {}
    n_ev = len(batches) * batch_size
    for _ in range(s.repeats):
        if is_cuda(device):
            torch.cuda.reset_peak_memory_stats()
        model_t = [timed(model.generate, device, b)[1] for b in batches]
        e2e_t = []
        for ids in batches_ids:
            sync(device)
            t0 = time.perf_counter()
            model.to_floats(model.generate(model.inputs(ids, batch_size)))
            sync(device)
            e2e_t.append(time.perf_counter() - t0)
        if scenario == "offline":
            tput = n_ev / sum(model_t)
            res.setdefault("throughput", []).append(tput)
            res.setdefault("throughput_e2e", []).append(n_ev / sum(e2e_t))
            res.setdefault("gpu_s_per_1k", []).append(1000.0 / tput * max(s.n_gpus, 1))
            if is_cuda(device):
                res.setdefault("peak_memory", []).append(torch.cuda.max_memory_allocated() / 2**30)
        else:
            ms, ms_e2e = 1e3 * np.asarray(model_t), 1e3 * np.asarray(e2e_t)
            for q in (50, 90, 99):
                res.setdefault(f"latency_p{q}", []).append(float(np.percentile(ms, q)))
            for q in (50, 99):
                res.setdefault(f"latency_e2e_p{q}", []).append(float(np.percentile(ms_e2e, q)))
    return res, t_warm


def gate_status(report: pd.DataFrame | None, model: str, gen: str, gate: Mapping | None) -> str:
    """`ok` if the quality row of the same generation config passes the gate, `invalid` if
    it fails, `missing` if there is no such quality row (quality run first)."""
    if not gate:
        return OK
    if report is None or report.empty:
        return MISSING
    q = overall(report[report["kind"] == "quality"])
    r = q[
        (q["model"] == model) & (q["metric"] == gate["metric"]) & (q["condition"] == f"gen={gen}")
    ]
    if r.empty:
        return MISSING
    r = r.iloc[0]
    if r["status"] != OK:
        return INVALID
    v = r["value"]
    if gate.get("max") is not None and v > gate["max"]:
        return INVALID
    if gate.get("min") is not None and v < gate["min"]:
        return INVALID
    return OK


def _row(model, metric, value, lower, upper, status, task_version, condition):
    """One perf row of `report.csv`."""
    return dict(
        model=model,
        metric=metric,
        bin_low=np.nan,
        bin_high=np.nan,
        value=value,
        lower=lower,
        upper=upper,
        status=status,
        kind="perf",
        task_version=task_version,
        condition=condition,
    )


def run_model(
    model_cfg: Mapping,
    settings: PerfSettings,
    rt: Runtime,
    events: np.ndarray,
    task_version: int,
    loaded: tuple[protocol.Model, float, dict] | None = None,
) -> tuple[list[dict], dict]:
    """Ungated perf rows and a meta block (hardware, model info) for one model. `loaded`
    reuses a model already loaded by `load_model` (the predict step)."""
    name = model_cfg["name"]
    hw = hardware(rt)
    gen = condition_hash(dict(model_cfg.get("generation") or {}))
    conditions = [
        (sc, b)
        for sc in settings.scenarios
        for b in ([1] if sc == "single" else settings.batch_sizes)
    ]
    if loaded is None and not model_cfg.get("model"):
        log.warning("%s: no `model` class configured; perf rows are missing", name)
        rows = [
            _row(
                name,
                m,
                np.nan,
                np.nan,
                np.nan,
                MISSING,
                task_version,
                perf_condition(gen, sc, b, hw),
            )
            for sc, b in conditions
            for m in (*COMMON, *METRICS[sc])
        ]
        return rows, dict(hardware=hw, model=None)

    model, load_s, info = loaded or load_model(model_cfg, rt)
    gen = condition_hash(dict(info.get("generation") or model_cfg.get("generation") or {}))
    first = model.inputs(events[: settings.batch_sizes[0]], settings.batch_sizes[0])
    _, compile_s = timed(model.generate, rt.device, first)

    rows = []
    for sc, b in conditions:
        cond = perf_condition(gen, sc, b, hw)
        res, warm = measure(model, events, sc, b, settings, rt.device)
        rows += [
            _row(name, "load_s", load_s, load_s, load_s, OK, task_version, cond),
            _row(name, "compile_s", compile_s, compile_s, compile_s, OK, task_version, cond),
            _row(name, "warmup_s", warm, warm, warm, OK, task_version, cond),
        ]
        for key in STATIC_INFO:
            v = info.get(key)
            st = OK if v is not None else MISSING
            v = np.nan if v is None else float(v)
            rows.append(_row(name, key, v, v, v, st, task_version, cond))
        for metric in METRICS[sc]:
            if metric not in res:
                rows.append(_row(name, metric, np.nan, np.nan, np.nan, MISSING, task_version, cond))
                continue
            mean, lo, hi, scatter = summary(res[metric])
            if scatter > MAX_SCATTER:
                log.warning(
                    "%s %s %s bs=%d: repeat scatter %.1f%% > 5%%",
                    name,
                    metric,
                    sc,
                    b,
                    100 * scatter,
                )
            rows.append(_row(name, metric, mean, lo, hi, OK, task_version, cond))
    return rows, dict(hardware=hw, model=info, load_s=load_s, compile_s=compile_s)


def apply_gate(perf_rows: pd.DataFrame, report: pd.DataFrame | None, gate: Mapping | None):
    """Perf rows with the quality gate applied: an `ok` row becomes `invalid` (gate failed)
    or `missing` (no quality row of the same generation config)."""
    out = perf_rows.copy()
    for (model, cond), idx in out.groupby(["model", "condition"]).groups.items():
        gen = cond.split(";")[0].removeprefix("gen=")
        status = gate_status(report, model, gen, gate)
        ok = out.loc[idx, "status"] == OK
        out.loc[ok[ok].index, "status"] = status
    return out


def merge_rows(report: pd.DataFrame | None, rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    """Append perf rows; rows with the same (model, metric, condition) are replaced."""
    new = pd.DataFrame(rows, columns=REPORT_COLUMNS)
    if report is None or report.empty:
        return new
    key = ["model", "metric", "condition"]
    old = report.merge(new[key], on=key, how="left", indicator=True)
    keep = report[(old["_merge"] == "left_only").to_numpy()]
    return pd.concat([keep, new], ignore_index=True)[REPORT_COLUMNS]
