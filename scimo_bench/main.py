"""Entry point: load the user's models file, compose the config, then run the steps
in-process or as the selected site's Snakemake workflow of jobs (`workflow.py`).

    ./run.sh site=raven run=cont_split_n5 models_file=nanohep_raven task=pflow \
        'models=[a,b]' steps=[predict,perf,score,plot] perf=single

Every call names its run (`run=<name>`); everything goes into `<paths.outputs>/<name>/`.
In-process, a call with `predict` or `score` into an existing run is refused unless `--force`
is given (it empties the run folder, keeping `predictions/` and, unless `perf` runs again,
the timings). The workflow of a cluster site reruns only what is missing or out of date in an
existing run; `--force` empties it the same way and reruns every requested step.
"""

from __future__ import annotations

import logging
import re
import shutil
import sys
from pathlib import Path

import hydra
from hydra_zen import store
from omegaconf import DictConfig, OmegaConf

from scimo_bench import io, jobs, plotter, steps, workflow
from scimo_bench.configs import MODEL_STEPS, STEPS, load_models_file

log = logging.getLogger("scimo_bench")

RUN_LOG = "run.log"
PREDICTIONS = "predictions"  # kept by --force: the models' outputs of the run
FORCE_FLAG = "--force"
RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class WarningCounter(logging.Handler):
    """Counts warnings (reported at the end of a run)."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.count = 0

    def emit(self, record):
        self.count += 1


def setup_logging(folder: Path, file: bool = True) -> WarningCounter:
    """Log to `run.log` in the run folder (if `file`) and to stderr."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    handlers = [logging.StreamHandler(sys.stderr)]
    if file:
        handlers.append(logging.FileHandler(folder / RUN_LOG))
    counter = WarningCounter()
    for h in handlers:
        h.setFormatter(fmt)
        root.addHandler(h)
    root.addHandler(counter)
    root.setLevel(logging.INFO)
    return counter


def ordered(names) -> list[str]:
    """Steps in their run order (predict, perf, score, plot)."""
    return [s for s in STEPS if s in set(names)]


def check_config(cfg: DictConfig) -> None:
    """Configuration errors stop the run; everything else only warns."""
    if OmegaConf.is_missing(cfg, "run"):
        raise ValueError("name the run: run=<name> (its folder is <paths.outputs>/<name>)")
    if not RUN_NAME.match(str(cfg.run)):
        raise ValueError(f"run name {cfg.run!r}: use letters, digits, '.', '_', '-' only")
    unknown = [s for s in cfg.steps if s not in STEPS]
    if unknown or not cfg.steps:
        raise ValueError(f"steps must be a non-empty list of {list(STEPS)}, got {list(cfg.steps)}")
    if "truth" in cfg.models:
        raise ValueError("'truth' is reserved and cannot be a model name")
    if cfg.only is not None and cfg.only not in cfg.models:
        raise ValueError(f"only={cfg.only}: not one of the models {list(cfg.models)}")
    if cfg.output.on_missing not in ("warn", "error"):
        raise ValueError("output.on_missing must be 'warn' or 'error'")
    cat = steps.dataset_catalogue(cfg)
    steps.task_metrics(cfg, cat)
    steps.plot_objects(cfg, cat)
    steps.check_params(cfg, cat)
    if "perf" in cfg.steps and cfg.perf is None:
        raise ValueError("step perf needs a perf config, e.g. perf=single")
    if "predict" in cfg.steps and not any(m.get("model") for m in cfg.models.values()):
        raise ValueError("step predict: none of the models has a `model` class to run")
    envs = {m.env for m in steps.selected(cfg).values() if m.get("model")}
    one_job = "perf" in cfg.steps or not jobs.should_submit(cfg.site, cfg.steps)
    if set(cfg.steps) & set(MODEL_STEPS) and len(envs) > 1 and one_job:
        raise ValueError(
            f"the models' classes need different envs {sorted(envs)}: they predict in one "
            "process (perf, or site=local); run them separately"
        )
    truth = OmegaConf.to_container(cfg.dataset.truth, resolve=False)
    if "score" in cfg.steps and "${paths.data}" in str(truth) and cfg.paths.data is None:
        raise ValueError(
            f"site '{cfg.site.name}' has no data root: set paths.data=<dir> "
            "(or dataset.truth.path=<file>)"
        )
    for name, mcfg in cfg.models.items():
        if mcfg.get("color") is not None:
            plotter.check_colour(mcfg.color, f"models.{name}.color")
    plotter.Style.from_cfg(OmegaConf.to_container(cfg.plots.style, resolve=True))
    jobs.check_site(cfg.site)


def run_dir(cfg: DictConfig) -> Path:
    """The run folder `output.dir` (default `<paths.outputs>/<run>`); `steps=[plot]` needs
    a scored run there."""
    folder = io.resolve_path(cfg.output.dir).resolve()
    if set(cfg.steps) == {"plot"} and not (folder / steps.REPORT).exists():
        raise FileNotFoundError(
            f"run '{cfg.run}' has no {steps.REPORT} in {folder}: score it first (steps=[score])"
        )
    return folder


def prepare_run_dir(cfg: DictConfig, folder: Path, workflow: bool = False) -> None:
    """A call with `predict` or `score` starts its run: an existing run folder is an
    error (in-process; a `workflow` updates it), unless `force` (--force), which empties it
    but keeps the models' outputs: `predictions/`, and the timings (`perf.csv`,
    `perf_meta_*.json`) unless this call runs `perf` again. Jobs skip this: the call that
    started them did it."""
    if jobs.in_job() or not {"predict", "score"} & set(cfg.steps):
        return
    if not folder.exists() or not any(folder.iterdir()):
        return
    if workflow and not cfg.force:
        return
    if not cfg.force:
        raise FileExistsError(
            f"run '{cfg.run}' exists: {folder}; choose another run=<name>, or overwrite it "
            f"with {FORCE_FLAG} (its {PREDICTIONS}/ and timings are kept)"
        )
    keep = {PREDICTIONS} | (set() if "perf" in cfg.steps else {steps.PERF})
    for p in folder.iterdir():
        if p.name in keep or ("perf" not in cfg.steps and p.name.startswith("perf_meta_")):
            continue
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()


def run(cfg: DictConfig, folder: Path) -> None:
    """Run the steps in-process."""
    todo = ordered(cfg.steps)
    if set(todo) & set(MODEL_STEPS):
        steps.run_models(cfg, folder)
    if "score" in todo:
        steps.score(cfg, folder)
    if "plot" in todo:
        steps.plot(cfg, folder)


@hydra.main(config_path=None, config_name="scimo_bench", version_base="1.3")
def bench(cfg: DictConfig) -> None:
    """Hydra entry: check the config, run or submit."""
    check_config(cfg)
    to_submit = jobs.should_submit(cfg.site, cfg.steps)
    if not to_submit:
        jobs.guard(cfg.site, cfg.steps)
    folder = run_dir(cfg)
    prepare_run_dir(cfg, folder, workflow=to_submit)
    folder.mkdir(parents=True, exist_ok=True)
    counter = setup_logging(folder, file=cfg.only is None)  # a model's job: its own log
    log.info("scimo_bench steps=%s commit=%s", list(cfg.steps), jobs.git_commit())
    log.info("call: %s", " ".join(sys.argv[1:]))
    log.info("output: %s", folder)
    if to_submit:
        workflow.run(cfg, folder)
        log.info("workflow done: %s", folder)
        return
    suffix = "" if "score" in cfg.steps else "." + "-".join(ordered(cfg.steps))
    suffix += f"-{cfg.only}" if cfg.only is not None else ""
    (folder / f"resolved_config{suffix}.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    run(cfg, folder)
    log.info("done: %s (%d warnings)", folder, counter.count)


def models_file_arg(argv: list[str]) -> str | None:
    """The `models_file=` value among the command-line overrides, if any."""
    for a in argv:
        key, _, value = a.partition("=")
        if key.lstrip("+") == "models_file" and value and value != "null":
            return value
    return None


def main() -> None:
    """`--force` becomes the override `force=true`; the user's models must be in the store
    before Hydra composes the config."""
    if FORCE_FLAG in sys.argv[1:]:
        sys.argv = [a for a in sys.argv if a != FORCE_FLAG] + ["force=true"]
    path = models_file_arg(sys.argv[1:])
    if path is not None:
        load_models_file(path)
    bench()


store.add_to_hydra_store(overwrite_ok=True)
