"""Where a call runs: in-process, or as jobs of the site's Snakemake workflow.

Site profiles are in `configs/sites.py` (paths, environments), their job resources in
`workflow/profiles/<site>/config.yaml`; `scimo_bench/workflow.py` runs the workflow. A job
runs from a frozen copy of the code taken at submit time, in a pixi environment of the repo
(or of the site). Nothing here computes physics.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import shlex
import shutil
import socket
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path

from hydra_zen import store
from omegaconf import OmegaConf

from scimo_bench import io
from scimo_bench.configs.sites import LOGIN_HOSTS, NVIDIA_SMI

SCHEDULERS = ("local", "slurm")
KINDS = ("cpu", "gpu")
ENV_JOB = "SCIMO_BENCH_JOB"
ENV_ALLOW = "SCIMO_BENCH_ALLOW_LOGIN"
VENDOR = "vendor"
PROFILES = Path("workflow") / "profiles"  # Snakemake profiles, one folder per site
SNAPSHOT_IGNORE = shutil.ignore_patterns(
    ".pixi",
    ".snakemake",
    "outputs",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "*.egg-info",
)


def git_commit(root: Path = io.REPO_ROOT) -> str:
    """The commit of the code (snapshot: from its COMMIT file); `-dirty` if modified."""

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        ).stdout.strip()

    sha = git("rev-parse", "--verify", "--quiet", "HEAD")
    snapshot = root / "COMMIT"  # written into job snapshots by `snapshot`
    if not sha and snapshot.exists():
        return snapshot.read_text().strip() + " (snapshot)"
    sha = sha or "no-commit"
    return sha + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


def hostname() -> str:
    """Short host name of this machine."""
    return socket.gethostname().split(".")[0]


def in_job() -> bool:
    """Inside a batch job or an allocation (`salloc`)."""
    return bool(os.environ.get(ENV_JOB) or os.environ.get("SLURM_JOB_ID"))


def login_site(host: str | None = None) -> str | None:
    """The site whose login nodes include `host`, over all site profiles."""
    host = host or hostname()
    for name, patterns in LOGIN_HOSTS.items():
        if any(re.fullmatch(p, host) for p in patterns):
            return name
    return None


def profile_dir(site: Mapping, root: Path = io.REPO_ROOT) -> Path:
    """The site's Snakemake profile: `site.profile` (default: the site's name) in
    `workflow/profiles/` of `root`, or a path."""
    name = site.get("profile") or site["name"]
    return Path(name) if "/" in str(name) else root / PROFILES / name


def check_site(site: Mapping) -> None:
    """A site profile's scheduler is valid and a cluster site has a Snakemake profile."""
    if site["scheduler"] not in SCHEDULERS:
        raise ValueError(f"site.scheduler must be one of {SCHEDULERS}, got {site['scheduler']!r}")
    if site["scheduler"] != "local" and not (profile_dir(site) / "config.yaml").is_file():
        raise ValueError(f"site '{site['name']}' has no Snakemake profile {profile_dir(site)}")


def inline(site: Mapping, steps) -> bool:
    """Steps that may run in-process even on a login node (redrawing plots)."""
    return set(steps) <= set(site["inline_steps"])


def should_submit(site: Mapping, steps) -> bool:
    """Whether these steps go to the scheduler instead of running here."""
    return site["scheduler"] != "local" and not in_job() and not inline(site, steps)


def guard(site: Mapping, steps, host: str | None = None) -> None:
    """Refuse to run steps in-process on a login node (outside a job)."""
    if in_job() or os.environ.get(ENV_ALLOW) or inline(site, steps):
        return
    host = host or hostname()
    name = login_site(host)
    if name:
        raise RuntimeError(
            f"{host} is a login node of site '{name}': add site={name} to submit "
            f"{list(steps)} as jobs ({ENV_ALLOW}=1 forces running here)"
        )


def env_root() -> Path:
    """The repo that holds the pixi environments: this one, or for code running from a
    snapshot the repo it was copied from (recorded in its `REPO` file)."""
    origin = io.REPO_ROOT / "REPO"
    return Path(origin.read_text().strip()) if origin.exists() else io.REPO_ROOT


def snapshot(snapshots: str | os.PathLike, bundle=()) -> Path:
    """Copy the code (no envs, outputs or git) to a new folder; `COMMIT` records its commit,
    `REPO` the repo with the environments. The `bundle` folders (relative to the repo) go
    to `vendor/`."""
    commit = git_commit()
    parent = Path(snapshots)
    parent.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = Path(tempfile.mkdtemp(prefix=f"{stamp}_{commit[:7]}_", dir=parent))
    shutil.copytree(io.REPO_ROOT, dest, ignore=SNAPSHOT_IGNORE, dirs_exist_ok=True)
    for src in bundle:
        src = (io.REPO_ROOT / src).resolve()
        shutil.copytree(src, dest / VENDOR / src.name, ignore=SNAPSHOT_IGNORE)
    (dest / "COMMIT").write_text(commit + "\n")
    (dest / "REPO").write_text(f"{env_root()}\n")
    return dest


def env_manifest(site: Mapping, env: str) -> tuple[Path, str]:
    """(pixi manifest, env name in it) of a job's environment `env` on this site."""
    spec = (site.get("envs") or {}).get(env)
    if spec:
        return Path(spec["manifest"]), spec["env"]
    return env_root() / "pixi.toml", env


def job_script(
    site: Mapping, kind: str, env: str, command: list[str], workdir: Path, name: str
) -> str:
    """The script a workflow job runs: activate the pixi env `env`, run `command` from the
    snapshot `workdir`. Resources come from the site's Snakemake profile."""
    q = shlex.quote
    pixi = site.get("pixi") or shutil.which("pixi") or "pixi"
    manifest, env_name = env_manifest(site, env)
    paths = [str(workdir / VENDOR)] if site.get("bundle") else []
    paths += ((site.get("envs") or {}).get(env) or {}).get("pythonpath", [])
    threads = "1" if env == "default" else "${SLURM_CPUS_PER_TASK:-1}"
    lines = [
        "#!/bin/bash -l",
        f"# {name}: written by scimo_bench.workflow for site '{site['name']}'",
        "# Before `set -u`: the env's activation scripts read unset variables.",
        f'eval "$({q(pixi)} shell-hook --manifest-path {q(str(manifest))} '
        f'-e {q(env_name)} --frozen --no-install)"',
        "set -euo pipefail",
        *site.get("setup", []),
        *(
            [f'export PYTHONPATH={q(":".join(paths))}"${{PYTHONPATH:+:$PYTHONPATH}}"']
            if paths
            else []
        ),
        f'export OPENBLAS_NUM_THREADS="${{OPENBLAS_NUM_THREADS:-{threads}}}"',
        f'export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-{threads}}}"',
        f"export {ENV_JOB}=1",
        f"cd {q(str(workdir))}",
        'echo "=== job ${SLURM_JOB_ID:-?} on ${SLURM_JOB_NODELIST:-$(hostname)} in $PWD, $(date)"',
    ]
    if kind == "gpu":
        lines.append(site.get("gpu_info") or NVIDIA_SMI)
    lines += [shlex.join(command), 'echo "=== done $(date)"']
    return "\n".join(lines) + "\n"


def check_env(site: Mapping, env: str) -> None:
    """The pixi environment a job needs is installed (checked before submitting)."""
    manifest, name = env_manifest(site, env)
    if not (manifest.parent / ".pixi" / "envs" / name).is_dir():
        flag = "" if name == "default" else f" -e {name}"
        where = "" if manifest == env_root() / "pixi.toml" else f" --manifest-path {manifest}"
        raise RuntimeError(
            f"pixi env '{env}' is not installed: pixi install{where}{flag} (README.md)"
        )


def check_folders(site: Mapping, *folders: Path) -> None:
    """Job folders must be on a filesystem the compute nodes see."""
    for p in folders:
        p = Path(p).resolve()
        if any(p.is_relative_to(d) for d in site.get("node_local") or []):
            raise ValueError(f"{p} is node-local; job folders must be on a shared filesystem")


def site_cfg(name: str) -> dict:
    """A site profile from the store, resolved."""
    return OmegaConf.to_container(OmegaConf.structured(store["site", name]), resolve=True)
