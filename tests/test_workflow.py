"""The Snakemake workflow of a cluster site, run on this machine (`site.profile=local`):
one predict job per model, perf, score, plot; reruns only what changed."""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scimo_bench import jobs, workflow
from scimo_bench.io import REPO_ROOT
from tests.conftest import run_bench, synthetic_overrides

pytestmark = pytest.mark.skipif(shutil.which("pixi") is None, reason="jobs activate pixi")

ALL = "steps=[predict,perf,score,plot]"


def overrides(contract_dir, tmp_path, *extra):
    """`dummy` (a file), `toy` and `toy2` (protocol models) on the synthetic truth, as the
    workflow of site raven with the local profile."""
    base = synthetic_overrides(contract_dir, tmp_path / "run", models=("ident",))
    ov = [o for o in base if not o.startswith("models=")] + ["models=[dummy,toy,toy2]"]
    for m in ("toy", "toy2"):
        ov += [
            f"models.{m}.predictions={tmp_path / m}",
            f"++models.{m}.options.truth={contract_dir / 'truth'}",
        ]
    return [
        *ov,
        "runtime=cpu",
        "site=raven",
        "site.profile=local",
        "site.node_local=[]",
        f"paths.snapshots={tmp_path / 's'}",
        *extra,
    ]


def stamps(run: Path) -> dict[str, int]:
    """Modification times of the steps' markers."""
    return {p.stem: p.stat().st_mtime_ns for p in (run / "status").glob("*.done")}


def test_workflow_runs_each_model_as_a_job(contract_dir, tmp_path):
    run = tmp_path / "run"
    run_bench(*overrides(contract_dir, tmp_path, ALL, "perf=smoke"))
    cfg = json.loads((run / workflow.CONFIG).read_text())
    assert cfg["steps"] == ["predict", "perf", "score", "plot"] and not cfg["gpu"]
    assert set(cfg["predict"]) == {"toy", "toy2"}  # dummy has no class: read as a file
    assert set(stamps(run)) == {"predict-toy", "predict-toy2", "perf", "score", "plot"}
    script = (run / "jobs" / "predict-toy.sh").read_text()
    assert "steps=[predict]" in script and script.endswith('only=toy\necho "=== done $(date)"\n')
    assert script.count("output.dir=") == 1 and f"output.dir={run}" in script
    snap = Path(script.split("\ncd ")[1].split("\n")[0])
    assert (snap / "COMMIT").exists() and (snap / "workflow" / "Snakefile").exists()
    assert (snap / "models_file.py").exists() and not (snap / ".pixi").exists()
    assert "toy: 400 events predicted" in (run / "logs" / "predict-toy.log").read_text()
    assert (run / "resolved_config.predict-toy2.yaml").exists()

    rep = pd.read_csv(run / "report.csv")
    overall = rep[rep.bin_low.isna() & (rep.metric == "response_median")]
    assert overall.set_index("model").value.to_dict() == pytest.approx(
        {"dummy": 1.0, "toy": 1.0, "toy2": 1.0}
    )
    perf_rows = rep[(rep.kind == "perf") & rep.value.notna()]
    assert set(perf_rows.model) == {"toy", "toy2"} and set(perf_rows.status) == {"ok"}
    assert (run / "plots" / "response_vs_e.png").exists()


def test_workflow_reruns_only_what_changed(contract_dir, tmp_path):
    run = tmp_path / "run"
    ov = overrides(contract_dir, tmp_path, "steps=[predict,score]")
    run_bench(*ov)
    first = stamps(run)
    proc = run_bench(*ov)  # an existing run: nothing to do
    assert stamps(run) == first and "Nothing to be done" in proc.stderr

    run_bench(*ov, "++models.toy2.options.scale=1.1")  # toy2 changed
    second = stamps(run)
    assert second["predict-toy"] == first["predict-toy"]
    assert second["predict-toy2"] > first["predict-toy2"] and second["score"] > first["score"]
    rep = pd.read_csv(run / "report.csv")
    resp = rep[rep.bin_low.isna() & (rep.metric == "response_median")].set_index("model").value
    assert resp["toy2"] == pytest.approx(1.1) and resp["toy"] == pytest.approx(1.0)

    run_bench(*ov, "++models.toy2.options.scale=1.1", "--force")  # everything again
    assert all(t > second[k] for k, t in stamps(run).items())


def test_failing_model_is_missing_not_fatal(contract_dir, tmp_path):
    run_bench(
        *overrides(contract_dir, tmp_path, "steps=[predict,score]"),
        "++models.toy2.options.report_generation.mode=sample",
    )
    rep = pd.read_csv(tmp_path / "run" / "report.csv")
    assert set(rep[rep.model == "toy2"].status) == {"missing"}
    assert set(rep[(rep.model == "toy") & rep.value.notna()].status) == {"ok"}
    assert "model failed" in (tmp_path / "run" / "logs" / "predict-toy2.log").read_text()


def test_models_in_different_envs_predict_but_not_perf(contract_dir, tmp_path):
    ov = overrides(contract_dir, tmp_path, "models.toy2.env=other")
    proc = run_bench(*ov, ALL, "perf=smoke", check=False)
    assert proc.returncode != 0 and "different envs" in proc.stderr
    proc = run_bench(*ov, "steps=[predict]", check=False)  # passes the check, then:
    assert proc.returncode != 0 and "pixi env 'other' is not installed" in proc.stderr


def test_workflow_cli_runs_a_command():
    # raven refuses node-local /tmp for jobs, so this test writes under the repo's outputs/
    (REPO_ROOT / "outputs").mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="test-workflow-", dir=REPO_ROOT / "outputs"))
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "scimo_bench.workflow", "raven", "--profile", "local"]
            + ["--log-dir", str(tmp / "l"), "--snapshots", str(tmp / "s"), "--"]
            + ["python", "-c", "print('hi from', __import__('os').getcwd())"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        (folder,) = (tmp / "l").iterdir()
        assert f"hi from {tmp / 's'}" in (folder / "logs" / "command.log").read_text()
    finally:
        shutil.rmtree(tmp)
    bad = subprocess.run(
        [sys.executable, "-m", "scimo_bench.workflow", "raven", "echo"], capture_output=True
    )
    assert bad.returncode != 0


@pytest.mark.parametrize("site", ["raven", "viper"])
def test_cluster_profiles(site, tmp_path):
    prof = yaml.safe_load((jobs.profile_dir(jobs.site_cfg(site)) / "config.yaml").read_text())
    assert prof["executor"] == "slurm"
    gpu = prof["set-resources"]
    assert gpu["predict"] == gpu["command_gpu"]
    assert gpu["perf"] == {**gpu["predict"], "runtime": 120}  # times every model in one job
    # the rules and resources the slurm executor would submit (dry run: nothing submitted)
    config = tmp_path / "workflow.json"
    config.write_text(
        json.dumps(
            dict(
                steps=list(workflow.STEPS),
                predict={"a": "1", "b": "2"},
                gpu=True,
                hash={"perf": "1", "score": "1", "plot": "1"},
            )
        )
    )
    proc = subprocess.run(
        [*workflow.snakemake_command(jobs.site_cfg(site), REPO_ROOT, config), "-n"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    out = proc.stdout + proc.stderr
    assert out.count("rule predict:") == 2 and "rule perf:" in out and "rule plot:" in out
    assert ("gres=gpu:a100:1" if site == "raven" else "slurm_partition=apu") in out
    if site == "raven":  # never the default partition `interactive`
        assert "slurm_partition=small" in out.split("rule score:")[1].split("rule ")[0]
        assert "slurm_partition=gpu," in out.split("rule perf:")[1].split("rule ")[0] + ","


def test_resources_of_a_call_add_to_the_profile(tmp_path):
    site = {
        **jobs.site_cfg("raven"),
        "set_resources": {"predict": {"slurm_partition": "gpudev", "runtime": 15}},
        "snakemake_args": ["--keep-going"],
    }
    config = tmp_path / "workflow.json"
    config.write_text(
        json.dumps(dict(steps=["predict", "perf"], predict={"a": "1"}, gpu=True, hash={}))
    )
    cmd = workflow.snakemake_command(site, REPO_ROOT, config, forcerun=["predict"])
    assert cmd[-3:] == ["--keep-going", "--forcerun", "predict"]
    proc = subprocess.run([*cmd, "-n"], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    out = proc.stdout + proc.stderr
    predict = out.split("rule predict:")[1].split("rule ")[0]
    assert "slurm_partition=gpudev" in predict and "runtime=15" in predict
    assert "gres=gpu:a100:1" in predict and "mem_mb=125000" in predict  # still the profile's
    perf = out.split("rule perf:")[1].split("rule ")[0]
    assert "gres=gpu:a100:1" in perf and "gpudev" not in perf and "runtime=120" in perf
