"""Where a call runs: in-process, or as a batch job of the site's scheduler.

Site profiles are in `configs/sites.py`. A job runs from a frozen copy of the code taken at
submit time, in the repo's pixi environment. Nothing here computes physics.

    python -m scimo_bench.jobs raven [--kind cpu] [--log-dir <dir>] -- pytest
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

from hydra_zen import store
from omegaconf import OmegaConf

from scimo_bench import io
from scimo_bench.configs.sites import LOGIN_HOSTS

log = logging.getLogger(__name__)

SCHEDULERS = ("local", "slurm")
KINDS = ("cpu", "gpu")
ENV_JOB = "SCIMO_BENCH_JOB"
ENV_ALLOW = "SCIMO_BENCH_ALLOW_LOGIN"
JOBS = "jobs.csv"
JOBS_COLUMNS = ["job_id", "name", "kind", "env", "after", "script", "snapshot", "submitted"]
SNAPSHOT_IGNORE = shutil.ignore_patterns(
    ".pixi", "outputs", ".git", "__pycache__", ".pytest_cache", ".ruff_cache", "*.egg-info"
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


def check_site(site: Mapping) -> None:
    """A site profile's scheduler and resources are valid (else config error)."""
    if site["scheduler"] not in SCHEDULERS:
        raise ValueError(f"site.scheduler must be one of {SCHEDULERS}, got {site['scheduler']!r}")
    if site["scheduler"] != "local" and not set(KINDS) <= set(site.get("resources") or {}):
        raise ValueError(f"site '{site['name']}' needs `resources` for {KINDS}")


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


def snapshot(snapshots: str | os.PathLike) -> Path:
    """Copy the code (no envs, outputs or git) to a new folder; `COMMIT` records its commit,
    `REPO` the repo with the environments."""
    commit = git_commit()
    parent = Path(snapshots)
    parent.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = Path(tempfile.mkdtemp(prefix=f"{stamp}_{commit[:7]}_", dir=parent))
    shutil.copytree(io.REPO_ROOT, dest, ignore=SNAPSHOT_IGNORE, dirs_exist_ok=True)
    (dest / "COMMIT").write_text(commit + "\n")
    (dest / "REPO").write_text(f"{env_root()}\n")
    return dest


def slurm_script(
    site: Mapping, kind: str, env: str, command: list[str], workdir: Path, log_path: Path, name: str
) -> str:
    """The sbatch script that runs `command` from the snapshot `workdir`."""
    q = shlex.quote
    pixi = shutil.which("pixi") or "pixi"
    threads = "1" if env == "default" else "${SLURM_CPUS_PER_TASK:-1}"
    lines = [
        "#!/bin/bash -l",
        f"#SBATCH --job-name=scimo-bench-{name}",
        f"#SBATCH --output={log_path}",
        *(f"#SBATCH --{k}={v}" for k, v in site["resources"][kind].items()),
        f"# written by scimo_bench.jobs for site '{site['name']}'",
        "# Before `set -u`: the env's activation scripts read unset variables.",
        f'eval "$({q(pixi)} shell-hook --manifest-path {q(str(env_root() / "pixi.toml"))} '
        f'-e {q(env)} --frozen --no-install)"',
        "set -euo pipefail",
        *site.get("setup", []),
        f'export OPENBLAS_NUM_THREADS="${{OPENBLAS_NUM_THREADS:-{threads}}}"',
        f'export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-{threads}}}"',
        f"export {ENV_JOB}=1",
        f"cd {q(str(workdir))}",
        'echo "=== job ${SLURM_JOB_ID:-?} on ${SLURM_JOB_NODELIST:-?} in $PWD, $(date)"',
    ]
    if kind == "gpu":
        lines.append(
            "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true"
        )
    lines += [f"srun {shlex.join(command)}", 'echo "=== done $(date)"']
    return "\n".join(lines) + "\n"


def check_env(env: str) -> None:
    """The pixi environment a job needs is installed (checked before submitting)."""
    if not (env_root() / ".pixi" / "envs" / env).is_dir():
        flag = "" if env == "default" else f" -e {env}"
        raise RuntimeError(f"pixi env '{env}' is not installed: pixi install{flag} (README.md)")


def submit(
    site: Mapping,
    kind: str,
    env: str,
    command: list[str],
    folder: Path,
    name: str,
    snap: Path,
    after: str | None = None,
) -> str:
    """Submit `command` as a job running in the code snapshot `snap`; script, log and
    `jobs.csv` go to `folder`. Returns the job id. `after` starts the job only once that job
    succeeded. With `site.wait` the call blocks until the job ends and fails if it fails."""
    check_env(env)
    folder = Path(folder).resolve()
    for p in (folder, Path(snap).resolve()):
        if any(p.is_relative_to(d) for d in site.get("node_local") or []):
            raise ValueError(f"{p} is node-local; job folders must be on a shared filesystem")
    folder.mkdir(parents=True, exist_ok=True)
    script = folder / f"job-{name}.sh"
    log_path = folder / f"job-{name}-%j.out"
    script.write_text(slurm_script(site, kind, env, command, snap, log_path, name))
    args = [
        "sbatch",
        "--parsable",
        *(["--wait"] if site.get("wait") else []),
        *([f"--dependency=afterok:{after}"] if after else []),
        str(script),
    ]
    log.info("submitting %s job (%s, env %s) to site '%s'", name, kind, env, site["name"])
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    job_id = proc.stdout.strip().split(";")[0]
    if not job_id:
        raise RuntimeError(f"sbatch failed ({proc.returncode}): {proc.stderr.strip()}")
    new = not (folder / JOBS).exists()
    with open(folder / JOBS, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(JOBS_COLUMNS)
        now = dt.datetime.now().isoformat(timespec="seconds")
        w.writerow([job_id, name, kind, env, after or "", script, snap, now])
    out = str(log_path).replace("%j", job_id)
    if proc.returncode != 0:
        raise RuntimeError(f"job {job_id} failed ({proc.returncode}); log: {out}")
    state = "finished" if site.get("wait") else "submitted"
    wait = f" (starts after job {after} succeeds)" if after else ""
    log.info("job %s %s%s; log: %s", job_id, state, wait, out)
    return job_id


def site_cfg(name: str) -> dict:
    """A site profile from the store, resolved."""
    return OmegaConf.to_container(OmegaConf.structured(store["site", name]), resolve=True)


def main(argv=None) -> int:
    """CLI: submit any command as a job of a site profile."""
    p = argparse.ArgumentParser(
        description="Run a command as a job of a site profile.",
        usage="python -m scimo_bench.jobs SITE [options] -- COMMAND ...",
    )
    p.add_argument("site", help="site profile, e.g. raven or local")
    p.add_argument("--kind", default="cpu", choices=KINDS)
    p.add_argument("--env", default="default", help="pixi environment of the job")
    p.add_argument("--log-dir", default=None, help="default: <site outputs>/jobs/<time>")
    p.add_argument("--wait", action="store_true", help="block until the job ends")
    p.add_argument("--snapshots", default=None, help="override the site's snapshot folder")
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--" not in argv:
        p.error("give the command after `--`")
    cut = argv.index("--")
    a, command = p.parse_args(argv[:cut]), argv[cut + 1 :]
    if not command:
        p.error("no command given")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    site = site_cfg(a.site)
    check_site(site)
    if site["scheduler"] == "local":
        guard(site, ["command"])
        return subprocess.run(command, cwd=io.REPO_ROOT, check=False).returncode
    site["wait"] = site["wait"] or a.wait
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = io.resolve_path(a.log_dir or f"{site['paths']['outputs']}/jobs") / stamp
    snap = snapshot(io.resolve_path(a.snapshots or site["paths"]["snapshots"]))
    submit(site, a.kind, a.env, command, folder, Path(command[0]).name, snap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
