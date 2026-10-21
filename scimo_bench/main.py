"""Entry point: load the user's models file, compose the config, then run the steps
in-process or submit them as jobs of the selected site.

    ./run.sh site=raven models_file=~/bench_models.py task=pflow 'models=[a,b]' \
        steps=[predict,perf,score,plot] perf=single
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra_zen import store
from omegaconf import DictConfig, OmegaConf

from scimo_bench import io, jobs, plotter, steps
from scimo_bench.configs import MODEL_STEPS, STEPS, load_models_file

log = logging.getLogger("scimo_bench")

RUN_LOG = "run.log"
MODELS_FILE = "models_file.py"  # the user's models file, copied into job snapshots


class WarningCounter(logging.Handler):
    """Counts warnings (reported at the end of a run)."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.count = 0

    def emit(self, record):
        self.count += 1


def setup_logging(folder: Path) -> WarningCounter:
    """Log to `run.log` in the run folder and to stderr."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(folder / RUN_LOG)
    sh = logging.StreamHandler(sys.stderr)
    counter = WarningCounter()
    for h in (fh, sh):
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
    unknown = [s for s in cfg.steps if s not in STEPS]
    if unknown or not cfg.steps:
        raise ValueError(f"steps must be a non-empty list of {list(STEPS)}, got {list(cfg.steps)}")
    if "truth" in cfg.models:
        raise ValueError("'truth' is reserved and cannot be a model name")
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
    envs = {m.env for m in cfg.models.values() if m.get("model")}
    if set(cfg.steps) & set(MODEL_STEPS) and len(envs) > 1:
        raise ValueError(f"the models' classes need different envs {sorted(envs)}; run separately")
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
    """The run folder: new for `predict`/`score`; otherwise (`perf`, `plot`) `output.dir`
    if it holds a report, else the latest run of the task that does."""
    folder = io.resolve_path(cfg.output.dir)
    if {"predict", "score"} & set(cfg.steps) or (folder / steps.REPORT).exists():
        return folder.resolve()
    task_root = io.resolve_path(cfg.paths.outputs) / cfg.task.name
    runs = sorted(p.parent for p in task_root.glob(f"*/{steps.REPORT}"))
    if not runs:
        raise FileNotFoundError(
            f"steps {list(cfg.steps)} need a scored run first: no {steps.REPORT} under {task_root}"
        )
    return runs[-1].resolve()


def job_groups(cfg: DictConfig) -> list[tuple[list[str], str, str]]:
    """(steps, resource kind, pixi env) per job: the model steps in one job on the runtime's
    device in the models' env, then score/plot in a CPU job."""
    todo = ordered(cfg.steps)
    groups = []
    model_steps = [s for s in todo if s in MODEL_STEPS]
    if model_steps:
        env = next((m.env for m in cfg.models.values() if m.get("model")), "default")
        kind = "gpu" if str(cfg.runtime.device).startswith("cuda") else "cpu"
        groups.append((model_steps, kind, env))
    rest = [s for s in todo if s not in MODEL_STEPS]
    if rest:
        groups.append((rest, "cpu", "default"))
    return groups


def job_overrides(cfg: DictConfig, folder: Path, todo: list[str], models_file: Path | None):
    """This call's overrides for a job: same choices, fixed folder, the job's steps."""
    fixed = ("output.dir", "steps", "models_file")
    keep = [
        o for o in HydraConfig.get().overrides.task if o.lstrip("+~").split("=")[0] not in fixed
    ]
    extra = [f"steps=[{','.join(todo)}]", f"output.dir={folder}"]
    if models_file is not None:
        extra.append(f"models_file={models_file}")
    return [*keep, *extra]


def submit(cfg: DictConfig, folder: Path) -> None:
    """Submit the steps as jobs of the site, writing into `folder` (fixed now); a later job
    starts only if the one before it succeeded."""
    site = OmegaConf.to_container(cfg.site, resolve=True)
    if not cfg.paths.snapshots:
        raise ValueError(f"site '{site['name']}' submits jobs: set paths.snapshots=<dir>")
    groups = job_groups(cfg)
    for _, _, env in groups:
        jobs.check_env(env)
    snap = jobs.snapshot(io.resolve_path(cfg.paths.snapshots))
    models_file = None
    if cfg.models_file:
        models_file = snap / MODELS_FILE
        shutil.copy(Path(cfg.models_file).expanduser(), models_file)
    after = None
    for todo, kind, env in groups:
        command = ["python", "-m", "scimo_bench", *job_overrides(cfg, folder, todo, models_file)]
        name = "-".join(todo)
        after = jobs.submit(site, kind, env, command, folder, name, snap=snap, after=after)


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
    folder.mkdir(parents=True, exist_ok=True)
    counter = setup_logging(folder)
    log.info("scimo_bench steps=%s commit=%s", list(cfg.steps), jobs.git_commit())
    log.info("call: %s", " ".join(sys.argv[1:]))
    log.info("output: %s", folder)
    if to_submit:
        submit(cfg, folder)
        return
    suffix = "" if "score" in cfg.steps else "." + "-".join(ordered(cfg.steps))
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
    """The user's models must be in the store before Hydra composes the config."""
    path = models_file_arg(sys.argv[1:])
    if path is not None:
        load_models_file(path)
    bench()


store.add_to_hydra_store(overwrite_ok=True)
