"""A call's steps as a Snakemake workflow of cluster jobs (`workflow/Snakefile`).

The call fixes the run folder, copies the code and the models file to a snapshot, writes one
job script per step (`jobs/<step>[-<model>].sh`: `predict` per model, `perf` for all models,
`score`, `plot`) and `workflow.json`, then runs Snakemake in the run folder with the site's
profile (`workflow/profiles/<site>`: job resources). The call blocks until the workflow
ends; run long ones in tmux. Snakemake reruns only what is missing or out of date.

    python -m scimo_bench.workflow raven [--kind gpu] [--env perf] -- pytest   # one job
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from scimo_bench import io, jobs
from scimo_bench.configs import MODEL_STEPS, STEPS, models_file_path

log = logging.getLogger("scimo_bench")

SNAKEFILE = Path("workflow") / "Snakefile"
CONFIG = "workflow.json"
JOB_DIR = "jobs"
MODELS_FILE = "models_file.py"  # the user's models file, copied into the snapshot
FIXED = ("output.dir", "steps", "models_file", "force", "only")  # set per job
# config entries each step reads: a change of their values reruns the step
# (`predict` also reads `models.<its model>`)
READS = {
    "predict": ("task", "dataset", "runtime"),
    "perf": ("models", "task", "dataset", "runtime", "perf"),
    "score": ("models", "task", "dataset", "cutflow", "output"),
    "plot": ("models", "task", "dataset", "cutflow", "output", "plots"),
}


def digest(full: Mapping, keys) -> str:
    """Short hash of these entries (dotted keys) of the resolved config."""
    part = {}
    for key in keys:
        node = full
        for k in key.split("."):
            node = node.get(k) if isinstance(node, Mapping) else None
        part[key] = node
    return hashlib.sha1(json.dumps(part, sort_keys=True, default=str).encode()).hexdigest()[:12]


def job_overrides(folder: Path, todo: list[str], models_file: Path | None) -> list[str]:
    """This call's overrides for a job: same choices, fixed folder, the job's steps."""
    keep = [
        o for o in HydraConfig.get().overrides.task if o.lstrip("+~").split("=")[0] not in FIXED
    ]
    extra = [f"steps=[{','.join(todo)}]", f"output.dir={folder}"]
    if models_file is not None:
        extra.append(f"models_file={models_file}")
    return [*keep, *extra]


def rule_names(cfg: DictConfig) -> dict[str, str]:
    """Step → Snakemake rule; model steps on a CPU are `<step>_cpu` (default resources)."""
    suffix = "" if str(cfg.runtime.device).startswith("cuda") else "_cpu"
    return {s: s + suffix if s in MODEL_STEPS else s for s in STEPS}


def plan(cfg: DictConfig, folder: Path, snap: Path, models_file: Path | None):
    """(workflow.json content, job scripts {name: text}) of this call."""
    site = OmegaConf.to_container(cfg.site, resolve=True)
    full = OmegaConf.to_container(cfg, resolve=True)
    gpu = str(cfg.runtime.device).startswith("cuda")
    kind = "gpu" if gpu else "cpu"
    todo = [s for s in STEPS if s in set(cfg.steps)]
    with_class = {n: m for n, m in full["models"].items() if m.get("model")}
    scripts, predict, hashes = {}, {}, {}

    def script(name, kind, env, step, *extra):
        cmd = ["python", "-m", "scimo_bench", *job_overrides(folder, [step], models_file), *extra]
        scripts[name] = jobs.job_script(site, kind, env, cmd, snap, name)

    for step in todo:
        if step == "predict":
            for name, m in with_class.items():
                predict[name] = digest(full, [f"models.{name}", *READS["predict"]])
                script(f"predict-{name}", kind, m["env"], "predict", f"only={name}")
            continue
        hashes[step] = digest(full, READS[step])
        if step == "perf":
            script("perf", kind, next((m["env"] for m in with_class.values()), "default"), step)
        else:
            script(step, "cpu", "default", step)
    return dict(steps=todo, predict=predict, hash=hashes, gpu=gpu), scripts


def needed_envs(cfg: DictConfig) -> set[str]:
    """The pixi environments the jobs of this call activate."""
    envs = {m.env for m in cfg.models.values() if m.get("model")} or {"default"}
    out = envs if set(cfg.steps) & set(MODEL_STEPS) else set()
    return out | ({"default"} if {"score", "plot"} & set(cfg.steps) else set())


def write(folder: Path, config: dict, scripts: Mapping[str, str]) -> Path:
    """The job scripts into `jobs/`, the workflow config into `workflow.json`."""
    (folder / JOB_DIR).mkdir(parents=True, exist_ok=True)
    for name, text in scripts.items():
        path = folder / JOB_DIR / f"{name}.sh"
        path.write_text(text)
        path.chmod(0o755)
    path = folder / CONFIG
    path.write_text(json.dumps(config, indent=2) + "\n")
    return path


def set_resources(site: Mapping, root: Path) -> list[str]:
    """`--set-resources` of the profile with the call's `site.set_resources` merged in (on
    the command line it replaces the profile's); none without overrides."""
    extra = site.get("set_resources") or {}
    if not extra:
        return []
    profile = yaml.safe_load((jobs.profile_dir(site, root) / "config.yaml").read_text())
    merged = {rule: dict(res) for rule, res in (profile.get("set-resources") or {}).items()}
    for rule, res in extra.items():
        merged.setdefault(rule, {}).update(res)
    values = [
        f"{rule}:{k}={v!r}" if isinstance(v, str) else f"{rule}:{k}={v}"
        for rule, res in merged.items()
        for k, v in res.items()
    ]
    return ["--set-resources", *values]


def snakemake_command(site: Mapping, snap: Path, config: Path, forcerun=()) -> list[str]:
    """Snakemake on the snapshot's Snakefile with the site's profile, the call's resource
    overrides (`site.set_resources`), extra options (`site.snakemake_args`) and this
    config."""
    return [
        sys.executable,
        "-m",
        "snakemake",
        "--snakefile",
        str(snap / SNAKEFILE),
        "--profile",
        str(jobs.profile_dir(site, snap)),
        "--configfile",
        str(config),
        *set_resources(site, snap),
        *site.get("snakemake_args", []),
        *(["--forcerun", *forcerun] if forcerun else []),
    ]


def execute(site: Mapping, folder: Path, command: list[str]) -> int:
    """Run Snakemake in `folder` (blocks until the workflow ends); its return code."""
    log.info("workflow on site '%s' (profile %s)", site["name"], jobs.profile_dir(site))
    log.info("snakemake: %s", " ".join(command[1:]))
    return subprocess.run(command, cwd=folder, check=False).returncode


def run(cfg: DictConfig, folder: Path) -> None:
    """Run the steps as the site's workflow, writing into `folder` (fixed now)."""
    site = OmegaConf.to_container(cfg.site, resolve=True)
    if not cfg.paths.snapshots:
        raise ValueError(f"site '{site['name']}' runs jobs: set paths.snapshots=<dir>")
    for env in sorted(needed_envs(cfg)):
        jobs.check_env(site, env)
    snapshots = io.resolve_path(cfg.paths.snapshots)
    jobs.check_folders(site, folder, snapshots)
    snap = jobs.snapshot(snapshots, site["bundle"])
    models_file = None
    if cfg.models_file:
        models_file = snap / MODELS_FILE
        shutil.copy(models_file_path(cfg.models_file), models_file)
    config, scripts = plan(cfg, folder, snap, models_file)
    path = write(folder, config, scripts)
    names = rule_names(cfg)
    forcerun = [names[s] for s in config["steps"]] if cfg.force else []
    code = execute(site, folder, snakemake_command(site, snap, path, forcerun))
    if code != 0:
        raise RuntimeError(f"workflow failed ({code}); job logs in {folder / 'logs'}")


def main(argv=None) -> int:
    """CLI: run any command as one job of a site's workflow."""
    p = argparse.ArgumentParser(
        description="Run a command as a job of a site's workflow.",
        usage="python -m scimo_bench.workflow SITE [options] -- COMMAND ...",
    )
    p.add_argument("site", help="site profile, e.g. raven or local")
    p.add_argument("--kind", default="cpu", choices=jobs.KINDS)
    p.add_argument("--env", default="default", help="pixi environment of the job")
    p.add_argument("--log-dir", default=None, help="default: <site outputs>/jobs/<time>")
    p.add_argument("--snapshots", default=None, help="override the site's snapshot folder")
    p.add_argument("--profile", default=None, help="override the site's Snakemake profile")
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--" not in argv:
        p.error("give the command after `--`")
    cut = argv.index("--")
    a, command = p.parse_args(argv[:cut]), argv[cut + 1 :]
    if not command:
        p.error("no command given")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    site = jobs.site_cfg(a.site)
    site["profile"] = a.profile or site["profile"]
    jobs.check_site(site)
    if site["scheduler"] == "local":
        jobs.guard(site, ["command"])
        return subprocess.run(command, cwd=io.REPO_ROOT, check=False).returncode
    jobs.check_env(site, a.env)
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = io.resolve_path(a.log_dir or f"{site['paths']['outputs']}/jobs") / stamp
    snapshots = io.resolve_path(a.snapshots or site["paths"]["snapshots"])
    jobs.check_folders(site, folder, snapshots)
    snap = jobs.snapshot(snapshots, site["bundle"])
    name = Path(command[0]).name
    script = jobs.job_script(site, a.kind, a.env, command, snap, name)
    config = dict(steps=[], predict={}, hash={}, gpu=a.kind == "gpu", command=True)
    path = write(folder, config, {"command": script})
    code = execute(site, folder, snakemake_command(site, snap, path))
    log.info("job log: %s", folder / "logs" / "command.log")
    return code


if __name__ == "__main__":
    sys.exit(main())
