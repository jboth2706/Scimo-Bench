"""The whole pipeline through the CLI on synthetic contract files."""

import hashlib

import awkward as ak
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from tests.conftest import MODELS_FILE, run_bench, synthetic_overrides

TOY = "tests.toy_model:ToyModel"


def digest(folder):
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((folder / "plots").iterdir())
    }


@pytest.fixture(scope="module")
def quality_run(contract_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("e2e") / "run"
    run_bench(*synthetic_overrides(contract_dir, out), "cutflow=central")
    return out


def test_output_folder(quality_run):
    names = {p.name for p in quality_run.iterdir()}
    assert {
        "resolved_config.yaml",
        "run.log",
        "report.csv",
        "rankings.csv",
        "cutflow.csv",
        "per_event.parquet",
        "models.csv",
        "plots",
    } <= names
    plots = {p.name for p in (quality_run / "plots").iterdir()}
    for stem in (
        "jet_e_response",
        "response_vs_pt",
        "efficiency_vs_pt",
        "ranking",
        "jet_e",
    ):
        assert {f"{stem}.png", f"{stem}.pdf"} <= plots
    cfg = OmegaConf.load(quality_run / "resolved_config.yaml")
    assert cfg.cutflow.steps[0].cut == "max_abs_eta" and cfg.cutflow.steps[0].on == "both"
    assert len(cfg.task.binnings.jet_pt) == 11  # resolvers resolved


def test_report_contents(quality_run):
    rep = pd.read_csv(quality_run / "report.csv")
    ov = rep[rep.bin_low.isna()].set_index(["model", "metric"])
    assert ov.loc[("dummy", "emd_median"), "value"] == pytest.approx(0.0, abs=1e-6)
    assert ov.loc[("dummy", "response_median"), "value"] == pytest.approx(1.0)
    assert ov.loc[("dummy", "efficiency"), "value"] == 1.0
    assert ov.loc[("second", "response_iqr"), "value"] > 0.02
    assert set(rep.kind) == {"quality"} and set(rep.task_version) == {1}
    assert rep.condition.str.startswith("gen=").all()
    rk = pd.read_csv(quality_run / "rankings.csv")
    best = rk[rk.metric == "mean_rank"].set_index("model")["rank"]
    assert best["dummy"] == 1
    cf = pd.read_csv(quality_run / "cutflow.csv")
    last = cf[cf.step == "min_particles"].set_index("model")["events"]
    assert last["truth"] == last["dummy"] == last["second"]
    pe = ak.from_parquet(quality_run / "per_event.parquet")
    assert set(np.unique(ak.to_numpy(pe.model))) == {"truth", "dummy", "second"}


def test_redraw_gives_identical_files(quality_run):
    before = digest(quality_run)
    run_bench(
        MODELS_FILE,
        "task=pflow",
        "dataset=synthetic",
        "models=[dummy,second]",
        "steps=[plot]",
        f"output.dir={quality_run}",
    )
    assert digest(quality_run) == before
    assert (quality_run / "resolved_config.plot.yaml").exists()


def test_same_seed_same_report(contract_dir, tmp_path, quality_run):
    out = tmp_path / "again"
    run_bench(*synthetic_overrides(contract_dir, out), "cutflow=central", "steps=[score]")
    a = (quality_run / "report.csv").read_bytes()
    assert (out / "report.csv").read_bytes() == a
    assert not (out / "plots").exists()


def test_missing_field_model_does_not_abort(contract_dir, tmp_path):
    out = tmp_path / "noe"
    run_bench(*synthetic_overrides(contract_dir, out, models=("ident", "noe")), "steps=[score]")
    rep = pd.read_csv(out / "report.csv")
    noe = rep[rep.model == "second"]
    assert (noe[noe.metric.str.startswith("response")].status == "missing").all()
    assert (noe[noe.metric == "emd_median"].status == "ok").all()
    assert "missing field 'particles.e'" in (out / "run.log").read_text()


def test_on_missing_error_stops(contract_dir, tmp_path):
    proc = run_bench(
        *synthetic_overrides(contract_dir, tmp_path / "x", models=("ident", "noe")),
        "output.on_missing=error",
        check=False,
    )
    assert proc.returncode != 0


def test_perf_and_pareto(contract_dir, tmp_path):
    out = tmp_path / "perf"
    # the perf generation config must equal the one in the prediction meta (same condition)
    gen = [
        "++models.dummy.generation.mode=synthetic",
        "++models.dummy.generation.multiplicity=model",
        "++models.second.generation.mode=smear",
        "++models.second.generation.sigma=0.1",
    ]
    ov = synthetic_overrides(contract_dir, out)
    run_bench(*ov, "steps=[score]")
    toy = [f"models.dummy.model={TOY}", f"models.second.model={TOY}"]
    run_bench(*ov, *gen, *toy, "steps=[perf]", "perf=smoke", "runtime=cpu")
    rep = pd.read_csv(out / "report.csv")
    p = rep[rep.kind == "perf"]
    assert set(p.model) == {"dummy", "second"}
    ok = p[p.metric == "latency_p50"]
    assert (ok.status == "ok").all(), ok
    # gate: response_iqr <= 0.25 holds for both synthetic models
    assert (out / "perf_meta_smoke.json").exists()
    run_bench(*ov, "steps=[plot]", "plots=pareto", "plots/pareto=emd_vs_flops")
    par = pd.read_csv(out / "pareto.csv")
    assert set(par.model) == {"dummy", "second"}
    # same FLOPs (same toy), dummy has EMD 0 -> only dummy on the front
    assert par.set_index("model").loc["dummy", "on_front"]
    assert not par.set_index("model").loc["second", "on_front"]
    assert (out / "plots" / "pareto_emd_vs_flops.png").exists()


def test_perf_finds_latest_run(contract_dir, tmp_path):
    """Without output.dir, steps perf/plot use the latest run of the task under paths.outputs."""
    root = tmp_path / "outputs"
    ov = [o for o in synthetic_overrides(contract_dir, root) if not o.startswith("output.dir")]
    run_bench(*ov, f"paths.outputs={root}", "steps=[score]")
    proc = run_bench(*ov, f"paths.outputs={root}", "steps=[plot]")
    runs = list((root / "pflow").iterdir())
    assert len(runs) == 1 and (runs[0] / "plots").exists()
    assert "output:" in proc.stderr


def test_plot_selection_from_the_command_line(contract_dir, tmp_path):
    out = tmp_path / "cfg"
    run_bench(
        *synthetic_overrides(contract_dir, out),
        "plots.extra=[jet_eta,particle_pt_response]",
        "plots.skip=[ranking]",
        "models.dummy.color=black",
        "plots.style.font_size=12",
    )
    names = {p.stem for p in (out / "plots").iterdir()}
    assert {"jet_eta", "particle_pt_response", "jet_e"} <= names and "ranking" not in names
    cfg = OmegaConf.load(out / "resolved_config.yaml")
    assert cfg.models.dummy.color == "black" and cfg.dataset.name == "synthetic"


def test_unknown_colour_is_a_config_error(contract_dir, tmp_path):
    proc = run_bench(
        *synthetic_overrides(contract_dir, tmp_path), "models.dummy.color=redd", check=False
    )
    assert proc.returncode != 0 and "colour 'redd'" in proc.stderr


def test_unknown_plot_name_is_a_config_error(contract_dir, tmp_path):
    proc = run_bench(
        *synthetic_overrides(contract_dir, tmp_path), "plots.extra=[nope]", check=False
    )
    assert proc.returncode != 0 and "plot 'nope'" in proc.stderr
