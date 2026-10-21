"""The four steps of a run, all working on files in the run folder.

- `predict` and `perf` run model code (protocol classes): predictions files, `perf.csv`.
- `score` reads truth and prediction files: `report.csv`, `rankings.csv`, `cutflow.csv`, ...
- `plot` draws from those files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
from hydra_zen import instantiate
from omegaconf import DictConfig, OmegaConf

from scimo_bench import catalogue, cutflow, io, perf, plotter, predict, scorer
from scimo_bench.schema import EVENT_ID, condition_hash, validate

log = logging.getLogger("scimo_bench")

REPORT = "report.csv"
RANKINGS = "rankings.csv"
CUTFLOW = "cutflow.csv"
PER_EVENT = "per_event.parquet"
MODELS = "models.csv"
PERF = "perf.csv"


def dataset_catalogue(cfg: DictConfig) -> catalogue.Catalogue:
    """The dataset's observables, comparisons and plots."""
    return catalogue.load(cfg.dataset.module)


def task_metrics(cfg: DictConfig, cat: catalogue.Catalogue) -> list[scorer.Metric]:
    """The task's metrics, instantiated (unknown names are config errors)."""
    metrics = list(instantiate(cfg.task.metrics))
    for m in metrics:
        cat.check_name(m.variable, f"metric {m.name}")
        if m.x:
            cat.check_name(m.x, f"metric {m.name} x")
    return metrics


def plot_names(cfg: DictConfig) -> list[str]:
    """`plots.draw` (default: the task's plots) + `plots.extra` − `plots.skip`, in order."""
    names = list(cfg.plots.draw or []) + list(cfg.plots.get("extra") or [])
    skip = set(cfg.plots.get("skip") or [])
    return [n for n in dict.fromkeys(names) if n not in skip]


def plot_objects(cfg: DictConfig, cat: catalogue.Catalogue) -> list:
    """The plots to draw: the dataset's plot functions or generic ones; Pareto plot last."""
    known = {**plotter.GENERIC_PLOTS, **cat.plots}
    objs = []
    for name in plot_names(cfg):
        if name not in known:
            raise ValueError(
                f"plot '{name}' is not a plot of dataset {cfg.dataset.name}; known: {sorted(known)}"
            )
        objs.append(known[name])
    if cfg.plots.get("pareto") is not None:
        objs.append(instantiate(cfg.plots.pareto))
    return objs


def check_params(cfg: DictConfig, cat: catalogue.Catalogue) -> None:
    """Task parameters must be keyword arguments of some comparison."""
    unknown = set(cfg.task.get("params") or {}) - cat.params()
    if unknown:
        raise ValueError(
            f"task.params {sorted(unknown)} are not parameters of any comparison of "
            f"{cfg.dataset.name}; known: {sorted(cat.params())}"
        )


def required_fields(cfg: DictConfig) -> set[str]:
    """Collections and fields the dataset requires from models."""
    out = set(cfg.dataset.get("event_fields") or [])
    for coll, fields in (cfg.dataset.get("collections") or {}).items():
        out |= {coll, *fields}
    return out


def missing(cfg: DictConfig, msg: str, *args) -> None:
    """A gap: warning, or error with `output.on_missing=error`."""
    if cfg.output.on_missing == "error":
        raise RuntimeError(msg % args)
    log.warning(msg, *args)


def event_list(cfg: DictConfig):
    """The task's event ids from its list file (None: all events)."""
    return io.read_event_list(cfg.task.events) if cfg.task.get("events") else None


def read_models(cfg: DictConfig) -> tuple[dict, list[scorer.ModelInfo], list[dict]]:
    """Read every model's predictions; a model that cannot be read is `missing`."""
    preds, infos, rows = {}, [], []
    for name, mcfg in cfg.models.items():
        path = mcfg.predictions
        gen = condition_hash(OmegaConf.to_container(mcfg.generation))
        info = scorer.ModelInfo(name, gen)
        meta = None
        try:
            arr, meta = io.read_predictions(path)
            files = io.parquet_files(path)
            size = sum(Path(f).stat().st_size for f in files)
            log.info(
                "%s: %d rows from %s (%d files, %.1f MB)",
                name,
                len(arr),
                path,
                len(files),
                size / 1e6,
            )
            for problem in validate(arr, meta, cfg.dataset):
                missing(cfg, "%s: %s", name, problem)
            if meta is not None:
                info.condition = meta.condition
                if meta.generation.get("multiplicity") == "oracle":
                    log.warning("%s: oracle multiplicity (labelled in plots)", name)
            if EVENT_ID not in arr.fields:
                raise ValueError(f"no '{EVENT_ID}'")
            preds[name] = arr
        except (OSError, ValueError) as e:
            missing(cfg, "%s: predictions unusable (%s); all its metrics are missing", name, e)
            info.available, info.reason = False, str(e)
        infos.append(info)
        mult = meta.generation.get("multiplicity") if meta is not None else None
        label = mcfg.get("label") or name
        rows.append(
            dict(
                model=name,
                label=label + (" (oracle mult.)" if mult == "oracle" else ""),
                family=(meta.model.get("family") if meta else None) or mcfg.get("family"),
                condition=f"gen={info.condition}",
                multiplicity=mult,
                available=info.available,
                predictions=str(path),
                meta=json.dumps(meta.to_dict(), default=str) if meta else None,
            )
        )
    return preds, infos, rows


def score(cfg: DictConfig, folder: Path) -> None:
    """Read truth and prediction files, cut, score; write report, rankings, cutflow."""
    cat = dataset_catalogue(cfg)
    metrics = task_metrics(cfg, cat)
    steps = instantiate(cfg.cutflow.steps) if cfg.cutflow is not None else []
    cutflow.check(steps)
    ids = event_list(cfg)
    if ids is not None:
        log.info("event list %s: %d events", cfg.task.events, len(ids))
    truth = read_truth(cfg, ids)

    preds, infos, model_rows = read_models(cfg)
    collection = cfg.task.collection
    joined = io.join(truth, preds, collection)
    joined = cutflow.apply(joined, steps)
    log.info("scoring %d events, %d models", len(joined.event_ids), len(joined.models))

    result = scorer.score(
        joined,
        infos,
        metrics,
        cat,
        params=OmegaConf.to_container(cfg.task.params),
        required=required_fields(cfg),
        n_boot=int(cfg.output.bootstrap.n),
        level=float(cfg.output.bootstrap.level),
        seed=int(cfg.output.seed),
        task_version=int(cfg.task.version),
    )
    rankings = scorer.rank(
        result.report,
        metrics,
        cfg.task.ranking.get("primary"),
        OmegaConf.to_container(cfg.task.get("normalization") or {}),
    )
    result.report.to_csv(folder / REPORT, index=False)
    rankings.to_csv(folder / RANKINGS, index=False)
    cutflow.to_frame(joined.cutflow).to_csv(folder / CUTFLOW, index=False)
    pd.DataFrame(model_rows).to_csv(folder / MODELS, index=False)
    ak.to_parquet(result.per_event, folder / PER_EVENT)
    for status, n in result.report["status"].value_counts().items():
        log.info("report: %d rows %s", n, status)
    apply_perf(cfg, folder)


def plot(cfg: DictConfig, folder: Path) -> None:
    """Draw the task's plots from the files of the run folder."""
    cat = dataset_catalogue(cfg)
    metrics = {m.name: m for m in task_metrics(cfg, cat)}
    plots = plot_objects(cfg, cat)
    report = pd.read_csv(folder / REPORT)
    rankings = pd.read_csv(folder / RANKINGS) if (folder / RANKINGS).exists() else None
    per_event = ak.from_parquet(folder / PER_EVENT) if (folder / PER_EVENT).exists() else None
    labels, overrides = {}, {}
    if (folder / MODELS).exists():
        labels = dict(pd.read_csv(folder / MODELS)[["model", "label"]].itertuples(index=False))
    for name, mcfg in cfg.models.items():
        labels.setdefault(name, mcfg.get("label") or name)
        overrides[name] = dict(color=mcfg.get("color"), linestyle=mcfg.get("linestyle"))
    models = list(cfg.models) or list(dict.fromkeys(report["model"]))
    pc = plotter.PlotContext(
        folder=folder,
        report=report[report["model"].isin(models)],
        rankings=None if rankings is None else rankings[rankings["model"].isin(models)],
        per_event=per_event,
        models=models,
        labels=labels,
        metrics=metrics,
        catalogue=cat,
        formats=list(cfg.output.formats),
        task=f"{cfg.task.name} v{cfg.task.version} · {cfg.dataset.name}",
        style=plotter.Style.from_cfg(OmegaConf.to_container(cfg.plots.style, resolve=True)),
        model_overrides=overrides,
    )
    paths = plotter.draw(pc, plots)
    log.info("wrote %d plot files to %s", len(paths), folder / "plots")


def read_truth(cfg: DictConfig, ids: np.ndarray | None = None):
    """The dataset's truth for these events (all if None), via its reader."""
    reader = instantiate(cfg.dataset.truth)
    truth = reader.read(ids)
    source = getattr(reader, "path", "")
    log.info("truth: %d events of dataset %s (%s)", len(truth), cfg.dataset.name, source)
    return truth


def task_events(cfg: DictConfig) -> np.ndarray:
    """The task's event ids: its event list, else every truth event."""
    ids = event_list(cfg)
    return ids if ids is not None else ak.to_numpy(read_truth(cfg)[EVENT_ID])


def save_perf(cfg: DictConfig, folder: Path, rows: list[dict], metas: dict, name: str) -> None:
    """Perf rows (ungated) into `perf.csv`, meta into `perf_meta_<name>.json`, then gate."""
    old = pd.read_csv(folder / PERF) if (folder / PERF).exists() else None
    perf.merge_rows(old, rows).to_csv(folder / PERF, index=False)
    meta_path = folder / f"perf_meta_{name}.json"
    meta_path.write_text(json.dumps(metas, indent=2, default=str))
    log.info("perf: %d rows in %s; meta in %s", len(rows), PERF, meta_path.name)
    apply_perf(cfg, folder)


def apply_perf(cfg: DictConfig, folder: Path) -> None:
    """Merge `perf.csv` into `report.csv` with the quality gate applied (needs both)."""
    if not (folder / PERF).exists() or not (folder / REPORT).exists():
        return
    report = pd.read_csv(folder / REPORT)
    gate = OmegaConf.to_container(cfg.task.quality_gate) if cfg.task.get("quality_gate") else None
    gated = perf.apply_gate(pd.read_csv(folder / PERF), report, gate)
    quality = report[report["kind"] == "quality"]
    perf.merge_rows(quality, gated).to_csv(folder / REPORT, index=False)
    for status, n in gated["status"].value_counts().items():
        log.info("perf rows %s: %d", status, n)


def run_models(cfg: DictConfig, folder: Path) -> None:
    """Steps `predict` and/or `perf`: each model with a protocol class is loaded once,
    writes its predictions, then is timed. One failing model does not stop the others."""
    do_predict, do_perf = "predict" in cfg.steps, "perf" in cfg.steps
    rt = perf.Runtime.from_cfg(OmegaConf.to_container(cfg.runtime))
    settings = perf.PerfSettings.from_cfg(OmegaConf.to_container(cfg.perf)) if do_perf else None
    ids = task_events(cfg)
    task = OmegaConf.to_container(cfg.task, resolve=True)
    dataset = OmegaConf.to_container(cfg.dataset, resolve=True)
    rows, metas = [], {}
    for name, mcfg in cfg.models.items():
        mcfg = OmegaConf.to_container(mcfg, resolve=True)
        if not mcfg.get("model"):
            log.info("%s: no `model` class; its prediction file is read as given", name)
            if do_perf:
                rows += perf.run_model(mcfg, settings, rt, ids, int(cfg.task.version))[0]
            continue
        try:
            loaded = None
            if do_predict:
                log.info("predict: %s on %d events (runtime=%s)", name, len(ids), rt.name)
                loaded = predict.run_model(mcfg, rt, ids, task, dataset)
            if do_perf:
                log.info("perf: %s (%s, runtime=%s)", name, settings.name, rt.name)
                r, metas[name] = perf.run_model(
                    mcfg, settings, rt, ids, int(cfg.task.version), loaded
                )
                rows += r
        except Exception:  # model code: one failing model must not stop the others
            log.exception("%s: model failed; its predictions / perf rows are missing", name)
    if do_perf:
        save_perf(cfg, folder, rows, metas, settings.name)
