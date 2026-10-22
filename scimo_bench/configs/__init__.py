"""hydra-zen config store. Importing this package fills the store.

Set per call: `run` (the run's name: its folder is `<paths.outputs>/<run>`), `site` (where
it runs, and the default `paths`), `task` (which brings its default `dataset`), `models`
(from your own `models_file`), `steps`. Everything else has
defaults that any override can change.
"""

import importlib.util
from pathlib import Path

import numpy as np
from hydra_zen import make_config, store
from omegaconf import MISSING, OmegaConf

from scimo_bench.configs import (  # noqa: F401
    cutflows,
    datasets,
    models,
    output,
    perf,
    plots,
    runtime,
    sites,
    tasks,
)


def register_resolvers() -> None:
    """`${linspace:a,b,n}` and `${logspace:a,b,n}` → lists of floats."""
    OmegaConf.register_new_resolver(
        "linspace", lambda a, b, n: np.linspace(float(a), float(b), int(n)).tolist(), replace=True
    )
    OmegaConf.register_new_resolver(
        "logspace", lambda a, b, n: np.logspace(float(a), float(b), int(n)).tolist(), replace=True
    )


register_resolvers()

STEPS = ("predict", "perf", "score", "plot")  # always run in this order
MODEL_STEPS = ("predict", "perf")  # run model code (GPU job on a cluster)

BenchConfig = make_config(
    hydra_defaults=[
        "_self_",
        {"task": MISSING},
        {"models": []},
        {"cutflow": None},
        {"plots": "default"},
        {"output": "default"},
        {"runtime": "gpu"},
        {"perf": None},
        {"site": "local"},
        {"override hydra/job_logging": "none"},
        {"override hydra/hydra_logging": "none"},
    ],
    run=MISSING,  # the run's name (required): its folder is <paths.outputs>/<run>
    force=False,  # --force: overwrite an existing run (its predictions/ are kept)
    task=None,
    dataset=None,  # set by the task's default (e.g. cocoa); `dataset=<name>` overrides
    models={},
    cutflow=None,
    plots=None,
    output=None,
    runtime=None,
    perf=None,
    site=None,
    # data root, run folders, job code snapshots; defaults from the site
    paths=dict(
        data="${site.paths.data}",
        outputs="${site.paths.outputs}",
        snapshots="${site.paths.snapshots}",
    ),
    models_file=None,
    steps=["score", "plot"],
    hydra=dict(run=dict(dir="."), output_subdir=None, job=dict(chdir=False)),
)

store(BenchConfig, name="scimo_bench")


MODELS_DIR = Path(__file__).resolve().parents[2] / "models"  # models files kept in the repo


def models_file_path(path: str | Path) -> Path:
    """A models file: a path, or the name of a file in the repo's `models/` folder
    (`nanohep_raven` → `models/nanohep_raven.py`)."""
    p = Path(path).expanduser()
    if not p.is_file() and p.parent == Path(".") and (MODELS_DIR / f"{p.stem}.py").is_file():
        p = MODELS_DIR / f"{p.stem}.py"
    if not p.is_file():
        known = sorted(f.stem for f in MODELS_DIR.glob("*.py"))
        raise FileNotFoundError(f"models_file {path}: no such file, nor one of {known}")
    return p.resolve()


def load_models_file(path: str | Path) -> Path:
    """Import a models file (`add_model(...)` calls) so its models join the store."""
    path = models_file_path(path)
    spec = importlib.util.spec_from_file_location(f"scimo_bench_models_{path.stem}", path)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    store.add_to_hydra_store(overwrite_ok=True)
    return path
