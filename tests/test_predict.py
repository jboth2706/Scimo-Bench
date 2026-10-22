"""The model steps: scimo-bench runs a protocol model, writes its predictions, times it,
and scores it."""

import json

import pandas as pd
import pytest

from scimo_bench import io
from scimo_bench.schema import validate
from tests import synthetic
from tests.conftest import run_bench, synthetic_overrides

ALL = "steps=[predict,perf,score,plot]"


def overrides(contract_dir, tmp_path, *extra):
    """The test model `toy` predicts the synthetic truth (pT, E × `options.scale`)."""
    base = synthetic_overrides(contract_dir, tmp_path / "run", models=())
    return [
        *(o for o in base if not o.startswith("models=")),
        "models=[toy]",
        f"models.toy.predictions={tmp_path / 'preds'}",
        f"++models.toy.options.truth={contract_dir / 'truth'}",
        "runtime=cpu",
        *extra,
    ]


def overall(run):
    rep = pd.read_csv(run / "report.csv")
    quality = rep[rep.bin_low.isna() & (rep.kind == "quality")]
    return rep, quality.set_index(["model", "metric"])


def test_all_steps_predict_score_and_time(contract_dir, tmp_path):
    run_bench(*overrides(contract_dir, tmp_path, ALL, "perf=smoke"))
    arr, meta = io.read_predictions(tmp_path / "preds")
    assert validate(arr, meta) == []
    assert len(arr) == 400 and meta.data["n_events"] == 400
    assert meta.code["producer"] == "tests.toy_model:ToyModel" and meta.extra["model_s"] > 0
    assert meta.hardware["device"] == "cpu"
    rep, ov = overall(tmp_path / "run")
    assert ov.loc[("toy", "response_median"), "value"] == pytest.approx(1.0)
    assert ov.loc[("toy", "response_iqr"), "value"] == pytest.approx(0.0, abs=1e-6)
    perf_rows = rep[(rep.kind == "perf") & rep.value.notna()]
    assert len(perf_rows) and set(perf_rows.status) == {"ok"}  # gate passed after scoring
    assert (tmp_path / "run" / "perf.csv").exists()
    assert (tmp_path / "run" / "plots" / "response_vs_pt.png").exists()


def test_failed_gate_and_known_shift(contract_dir, tmp_path):
    run_bench(
        *overrides(contract_dir, tmp_path, ALL, "perf=smoke"),
        "++models.toy.options.scale=1.1",
        "task.quality_gate.max=0.0",
    )
    rep, ov = overall(tmp_path / "run")
    assert ov.loc[("toy", "response_median"), "value"] == pytest.approx(1.1)
    assert set(rep[(rep.kind == "perf") & rep.value.notna()].status) == {"invalid"}


def test_predict_only_writes_the_file(contract_dir, tmp_path):
    run_bench(*overrides(contract_dir, tmp_path, "steps=[predict]"))
    meta = json.loads((tmp_path / "preds" / "meta.json").read_text())
    assert meta["data"]["task"] == "pflow" and len(meta["data"]["event_list_hash"]) == 10
    assert meta["model"]["family"] == "regression" and meta["model"]["n_params"] == 1024
    assert not (tmp_path / "run" / "report.csv").exists()
    assert not (tmp_path / "run" / "perf.csv").exists()


def test_failing_model_leaves_no_stale_file(contract_dir, tmp_path, truth):
    # an older prediction file of this model, then a run where the model fails
    synthetic.write(tmp_path, "preds", synthetic.identical(truth))
    run_bench(
        *overrides(contract_dir, tmp_path, "steps=[predict,score]"),
        "++models.toy.options.report_generation.mode=sample",
    )
    assert not list((tmp_path / "preds").glob("*.parquet"))
    log = (tmp_path / "run" / "run.log").read_text()
    assert "must match" in log and "model failed" in log
    rep, _ = overall(tmp_path / "run")
    assert set(rep[rep.model == "toy"].status) == {"missing"}


def test_predict_needs_a_model_class(contract_dir, tmp_path):
    ov = synthetic_overrides(contract_dir, tmp_path, models=("ident",))  # dummy: files only
    proc = run_bench(*ov, "steps=[predict]", check=False)
    assert proc.returncode != 0 and "`model` class" in proc.stderr


def test_cluster_chains_a_model_job_and_a_scoring_job(contract_dir, tmp_path, fake_slurm):
    run = tmp_path / "run"
    run_bench(
        *overrides(contract_dir, tmp_path, ALL, "perf=smoke", "site=raven"),
        "site.node_local=[]",
        f"paths.snapshots={tmp_path}/s",
        env=fake_slurm,
    )
    sub = pd.read_csv(run / "jobs.csv", dtype=str)
    assert list(sub.name) == ["predict-perf", "score-plot"]
    assert sub.after.fillna("").tolist() == ["", "12345"]
    assert sub.snapshot.nunique() == 1 and set(sub.kind) == {"cpu"}  # runtime=cpu
    calls = (tmp_path / "sbatch.log").read_text().splitlines()
    assert "--dependency" not in calls[0] and "--dependency=afterok:12345" in calls[1]
    assert "steps=[predict,perf]" in (run / "job-predict-perf.sh").read_text()
    assert "steps=[score,plot]" in (run / "job-score-plot.sh").read_text()
    assert not (tmp_path / "preds").exists()  # nothing ran yet
