"""Site profiles, the login-node guard and job submission (fake `sbatch`/`srun`)."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from scimo_bench import jobs
from scimo_bench.io import REPO_ROOT
from tests.conftest import JOB_VARS, MODELS_FILE, run_bench, synthetic_overrides

RAVEN = jobs.site_cfg("raven")
LOCAL = jobs.site_cfg("local")


@pytest.fixture
def outside_job(monkeypatch):
    for k in JOB_VARS:
        monkeypatch.delenv(k, raising=False)


def test_guard_refuses_login_nodes(outside_job):
    with pytest.raises(RuntimeError, match="add site=raven"):
        jobs.guard(LOCAL, ["score", "plot"], host="raven01")
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


def test_slurm_script():
    cmd = ["python", "-m", "scimo_bench", "models.dummy.label=a b"]
    log = Path("/out/job-score-%j.out")
    cpu = jobs.slurm_script(RAVEN, "cpu", "default", cmd, Path("/snap"), log, "score")
    assert "#SBATCH --cpus-per-task=4" in cpu and "gres" not in cpu
    assert "#SBATCH --output=/out/job-score-%j.out" in cpu
    assert "-e default --frozen --no-install" in cpu and "cd /snap" in cpu
    assert "srun python -m scimo_bench 'models.dummy.label=a b'" in cpu
    gpu = jobs.slurm_script(RAVEN, "gpu", "perf", cmd, Path("/snap"), log, "perf")
    assert "#SBATCH --gres=gpu:a100:1" in gpu and "-e perf" in gpu and "nvidia-smi" in gpu


def test_bad_site_is_a_config_error():
    with pytest.raises(ValueError, match="scheduler"):
        jobs.check_site({**RAVEN, "scheduler": "pbs"})
    with pytest.raises(ValueError, match="resources"):
        jobs.check_site({**RAVEN, "resources": {}})
    proc = run_bench("task=pflow", "site=raven", "site.scheduler=pbs", check=False)
    assert proc.returncode != 0 and "site.scheduler" in proc.stderr


def test_missing_env_fails_before_submitting(tmp_path):
    with pytest.raises(RuntimeError, match="not installed"):
        jobs.submit(RAVEN, "gpu", "no_such_env", ["true"], tmp_path, "x", tmp_path)
    assert not (tmp_path / "job-x.sh").exists()


def test_node_local_folders_are_refused(tmp_path):
    with pytest.raises(ValueError, match="node-local"):
        jobs.submit(
            {**RAVEN, "node_local": [str(tmp_path)]},
            "cpu",
            "default",
            ["true"],
            tmp_path / "f",
            "x",
            tmp_path / "s",
        )


@pytest.mark.skipif(shutil.which("pixi") is None, reason="the job script activates pixi")
def test_submit_then_run_the_job(contract_dir, tmp_path, fake_slurm):
    out = tmp_path / "run"
    ov = [
        *synthetic_overrides(contract_dir, out),
        "site=raven",
        "site.node_local=[]",
        f"paths.snapshots={tmp_path}/s",
    ]
    run_bench(*ov, "steps=[score]", env=fake_slurm)
    # submitted, not run
    assert not (out / "report.csv").exists()
    sub = pd.read_csv(out / "jobs.csv")
    assert list(sub.job_id) == [12345] and sub.kind[0] == "cpu" and sub.env[0] == "default"
    snap = Path(sub.snapshot[0])
    assert (snap / "COMMIT").exists() and (snap / "scimo_bench" / "main.py").exists()
    assert not (snap / ".pixi").exists() and not (snap / "outputs").exists()
    assert (snap / "models_file.py").exists()  # the user's models file, frozen too
    assert "--parsable" in (tmp_path / "sbatch.log").read_text()
    script = (out / "job-score.sh").read_text()
    assert script.count("output.dir=") == 1 and f"output.dir={out}" in script
    assert f"models_file={snap}/models_file.py" in script and "steps=[score]" in script

    # what the scheduler would do: run the script; it runs inline from the snapshot
    proc = subprocess.run(["bash", out / "job-score.sh"], env=fake_slurm, capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode()
    assert (out / "report.csv").exists()
    assert "(snapshot)" in (out / "run.log").read_text()

    # plot is an inline step: no new job
    run_bench(*ov, "steps=[plot]", env=fake_slurm)
    assert (out / "plots").is_dir() and len(pd.read_csv(out / "jobs.csv")) == 1


def test_jobs_cli(fake_slurm):
    # raven refuses node-local /tmp for jobs, so this test writes under the repo's outputs/
    (REPO_ROOT / "outputs").mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="test-jobs-", dir=REPO_ROOT / "outputs"))
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "scimo_bench.jobs", "raven", "--log-dir", str(tmp / "l")]
            + ["--snapshots", str(tmp / "s"), "--", "echo", "hi"],
            cwd=REPO_ROOT,
            env=fake_slurm,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        (folder,) = (tmp / "l").iterdir()
        assert "srun echo hi" in (folder / "job-echo.sh").read_text()
    finally:
        shutil.rmtree(tmp)
    bad = subprocess.run(
        [sys.executable, "-m", "scimo_bench.jobs", "raven", "echo"], cwd=REPO_ROOT, check=False
    )
    assert bad.returncode != 0
