"""Site profiles, the login-node guard, job scripts and code snapshots."""

import os
from pathlib import Path

import pytest

from scimo_bench import jobs
from scimo_bench.io import REPO_ROOT
from tests.conftest import JOB_VARS, MODELS_FILE, run_bench

RAVEN = jobs.site_cfg("raven")
VIPER = jobs.site_cfg("viper")
LOCAL = jobs.site_cfg("local")


@pytest.fixture
def outside_job(monkeypatch):
    for k in JOB_VARS:
        monkeypatch.delenv(k, raising=False)


def test_guard_refuses_login_nodes(outside_job):
    with pytest.raises(RuntimeError, match="add site=raven"):
        jobs.guard(LOCAL, ["score", "plot"], host="raven01")
    with pytest.raises(RuntimeError, match="add site=viper"):
        jobs.guard(LOCAL, ["score"], host="viper01")
    jobs.guard(LOCAL, ["plot"], host="raven01")  # redraws may run here
    jobs.guard(LOCAL, ["score"], host="ravc2001")  # compute node
    jobs.guard(LOCAL, ["score"], host="laptop")


def test_guard_passes_inside_a_job(outside_job, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    jobs.guard(LOCAL, ["score"], host="raven01")


def test_cli_refuses_login_node(monkeypatch):
    env = {k: v for k, v in os.environ.items() if k not in JOB_VARS}
    if jobs.login_site() is None:
        pytest.skip("not on a known login node")
    proc = run_bench(
        MODELS_FILE, "task=pflow", "models=[dummy]", "paths.data=/x", check=False, env=env
    )
    assert proc.returncode != 0 and "is a login node" in proc.stderr


def test_should_submit(outside_job, monkeypatch):
    assert jobs.should_submit(RAVEN, ["score", "plot"]) and jobs.should_submit(RAVEN, ["perf"])
    assert not jobs.should_submit(RAVEN, ["plot"])
    assert not jobs.should_submit(LOCAL, ["score"])
    monkeypatch.setenv(jobs.ENV_JOB, "1")
    assert not jobs.should_submit(RAVEN, ["score"])


def test_job_script():
    cmd = ["python", "-m", "scimo_bench", "models.dummy.label=a b"]
    cpu = jobs.job_script(RAVEN, "cpu", "default", cmd, Path("/snap"), "score")
    assert "#SBATCH" not in cpu and "srun" not in cpu  # resources: the Snakemake profile
    assert "-e default --frozen --no-install" in cpu and "cd /snap" in cpu
    assert "\npython -m scimo_bench 'models.dummy.label=a b'\n" in cpu
    assert "SCIMO_BENCH_JOB=1" in cpu and "nvidia-smi" not in cpu
    gpu = jobs.job_script(RAVEN, "gpu", "perf", cmd, Path("/snap"), "perf")
    assert "-e perf" in gpu and "nvidia-smi" in gpu


def test_job_script_with_site_envs():
    cmd = ["python", "-m", "scimo_bench"]
    gpu = jobs.job_script(VIPER, "gpu", "perf", cmd, Path("/snap"), "perf")
    rocm = VIPER["envs"]["perf"]
    assert f"--manifest-path {rocm['manifest']} -e default" in gpu
    assert f"export PYTHONPATH=/snap/vendor:{rocm['pythonpath'][0]}" in gpu
    assert VIPER["pixi"] in gpu and "rocm-smi" in gpu and "nvidia-smi" not in gpu
    cpu = jobs.job_script(VIPER, "cpu", "default", cmd, Path("/snap"), "score")
    assert VIPER["envs"]["default"]["manifest"] in cpu and "rocm-extras" not in cpu
    assert "export PYTHONPATH=/snap/vendor" in cpu
    # raven: no bundle, no extra paths
    assert "PYTHONPATH" not in jobs.job_script(RAVEN, "cpu", "default", cmd, Path("/s"), "x")


def test_site_env_check_and_bundle(tmp_path):
    (tmp_path / "env" / ".pixi" / "envs" / "rocm").mkdir(parents=True)
    site = {**RAVEN, "envs": {"perf": {"manifest": str(tmp_path / "env/pixi.toml"), "env": "rocm"}}}
    jobs.check_env(site, "perf")
    with pytest.raises(RuntimeError, match=f"--manifest-path {tmp_path}/other"):
        jobs.check_env(
            {**site, "envs": {"perf": {"manifest": f"{tmp_path}/other/pixi.toml", "env": "x"}}},
            "perf",
        )
    pkg = tmp_path / "producer" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "mod.py").write_text("X = 1\n")
    snap = jobs.snapshot(tmp_path / "s", bundle=[str(pkg)])
    assert (snap / "vendor" / "pkg" / "mod.py").exists()


def test_bad_site_is_a_config_error():
    with pytest.raises(ValueError, match="scheduler"):
        jobs.check_site({**RAVEN, "scheduler": "pbs"})
    with pytest.raises(ValueError, match="no Snakemake profile"):
        jobs.check_site({**RAVEN, "profile": "nowhere"})
    jobs.check_site({**RAVEN, "profile": str(REPO_ROOT / "workflow/profiles/local")})
    proc = run_bench("task=pflow", "site=raven", "site.scheduler=pbs", check=False)
    assert proc.returncode != 0 and "site.scheduler" in proc.stderr


def test_node_local_folders_are_refused(tmp_path):
    jobs.check_folders(RAVEN, Path("/ptmp/someone/run"))
    with pytest.raises(ValueError, match="node-local"):
        jobs.check_folders({**RAVEN, "node_local": [str(tmp_path)]}, tmp_path / "f")
