import pytest
from hydra import compose, initialize
from hydra_zen import builds, instantiate
from omegaconf import OmegaConf

from scimo_bench import catalogue, plotter
from scimo_bench import steps as bench_steps
from scimo_bench.configs import load_models_file
from scimo_bench.configs.models import add_model
from scimo_bench.cutflow import CutStep
from scimo_bench.io import REPO_ROOT
from scimo_bench.plotter import ParetoPlot
from scimo_bench.scorer import IQR, Median, Metric
from tests.conftest import MODELS_FILE, run_bench

load_models_file(REPO_ROOT / "tests" / "models.py")


def cfg(*overrides):
    with initialize(version_base="1.3", config_path=None):
        return compose(config_name="scimo_bench", overrides=list(overrides))


def test_defaults_compose():
    c = cfg("task=pflow")
    assert list(c.steps) == ["score", "plot"]
    assert c.site.name == "local" and c.runtime.name == "gpu"
    assert c.models == {}
    assert c.cutflow is None and c.perf is None
    assert c.output.bootstrap.n == 1000 and c.output.bootstrap.level == 0.95
    assert list(c.output.formats) == ["pdf"]


def test_task_is_required():
    with pytest.raises(Exception, match="task"):
        OmegaConf.to_container(cfg(), resolve=True, throw_on_missing=True)


def test_several_models_in_order():
    c = cfg("task=pflow", "models=[nano,dummy]")
    assert list(c.models) == ["nano", "dummy"]
    assert c.models.nano.family == "token_ar" and c.models.nano.env == "perf"
    assert c.models.nano.model == "examples.nanohep_model:NanoHEP"
    assert c.models.nano.generation.mode == "argmax"
    assert c.models.dummy.model is None and c.models.dummy.label == "dummy"


def test_models_file_by_name():
    from scimo_bench.configs import MODELS_DIR, models_file_path

    assert models_file_path("nanohep_raven") == MODELS_DIR / "nanohep_raven.py"
    assert models_file_path("tests/models.py").name == "models.py"  # a path stays a path
    with pytest.raises(FileNotFoundError, match="nanohep_raven"):
        models_file_path("nope")


def test_paths_come_from_the_site(monkeypatch):
    monkeypatch.setenv("USER", "someone")
    c = cfg("task=pflow", "models=[dummy]", "site=raven", "run=sweep")
    assert c.output.dir == "/ptmp/someone/bench/outputs/sweep"
    assert c.models.dummy.predictions == "/ptmp/someone/bench/outputs/sweep/predictions/dummy"
    assert c.dataset.name == "cocoa" and c.dataset.truth.path.startswith("/ptmp/someone/COCOA/")
    c = cfg("task=pflow", "site=raven", "paths.data=/elsewhere")
    assert c.dataset.truth.path.startswith("/elsewhere/COCOA/")
    assert cfg("task=pflow").paths.data is None


def test_add_model_rejects_typos():
    with pytest.raises(TypeError, match="checkpont"):
        add_model("x", checkpont="/a")
    with pytest.raises(ValueError, match="unknown type"):
        add_model("x", type="nope")


def test_linspace_resolver():
    c = OmegaConf.create({"b": "${linspace:0,2,51}", "l": "${logspace:0,2,3}"})
    b = c.b
    assert len(b) == 51 and all(isinstance(v, float) for v in b)
    assert b[0] == 0.0 and b[-1] == 2.0
    assert list(c.l) == pytest.approx([1.0, 10.0, 100.0])


def test_misspelled_builds_argument_fails_at_compose_time():
    with pytest.raises(TypeError, match="nmae"):
        builds(Median, nmae="response_iqr", variable="jet_e_response")
    with pytest.raises(TypeError, match="valeu"):
        builds(CutStep, cut="min_pt", valeu=1.0)


def test_task_metrics_and_plots_instantiate():
    c = cfg("task=pflow", "cutflow=preselection", "plots=pareto")
    cat = bench_steps.dataset_catalogue(c)
    metrics = bench_steps.task_metrics(c, cat)
    assert all(isinstance(m, Metric) for m in metrics)
    by_name = {m.name: m for m in metrics}
    assert isinstance(by_name["response_iqr"], IQR)
    assert by_name["response_iqr"].binning == pytest.approx(list(range(0, 201, 20)))
    assert by_name["response_median"].better == {"closest_to": 1.0}
    plots = bench_steps.plot_objects(c, cat)
    assert plots[0] is cat.plots["jet_e_response"] and plotter.ranking in plots
    assert isinstance(plots[-1], ParetoPlot) and plots[-1].x == "latency_p50"
    steps = instantiate(c.cutflow.steps)
    assert [s.cut for s in steps] == ["max_abs_eta", "min_pt", "min_particles"]


def test_pareto_subgroup_override():
    c = cfg("task=pflow", "plots=pareto", "plots/pareto=iqr_vs_flops")
    assert c.plots.pareto.x == "flops_per_event"


def test_perf_configs():
    c = cfg("task=pflow", "perf=offline")
    assert list(c.perf.batch_sizes) == [1, 64, 256]
    assert c.perf.warmup == 20 and c.perf.repeats == 5
    assert cfg("task=pflow", "runtime=cpu").runtime.precision == "fp32"


def test_dummy_run_writes_folder(tmp_path, contract_dir):
    out = tmp_path / "run"
    proc = run_bench(
        MODELS_FILE,
        "task=pflow",
        "models=[dummy]",
        f"output.dir={out}",
        "task.events=null",
        "dataset=synthetic",
        f"dataset.truth.path={contract_dir / 'truth'}",
        "output.bootstrap.n=0",
    )
    assert proc.returncode == 0
    assert (out / "resolved_config.yaml").exists()
    resolved = OmegaConf.load(out / "resolved_config.yaml")
    assert resolved.task.name == "pflow" and "dummy" in resolved.models
    log = (out / "run.log").read_text()
    assert "models=[dummy]" in log and "commit=" in log
    assert "dummy: predictions unusable" in log


def test_bad_step_is_a_config_error(tmp_path):
    proc = run_bench("task=pflow", "steps=[nope]", f"output.dir={tmp_path}", check=False)
    assert proc.returncode != 0
    assert "steps must be" in proc.stderr


def test_local_site_needs_a_data_root(tmp_path):
    proc = run_bench(
        MODELS_FILE, "task=pflow", "models=[dummy]", f"output.dir={tmp_path}", check=False
    )
    assert proc.returncode != 0 and "set paths.data" in proc.stderr


def test_dataset_brings_its_reader_and_catalogue():
    c = cfg("task=pflow")
    assert c.dataset.name == "cocoa" and c.dataset.module == "scimo_bench.datasets.cocoa"
    assert c.dataset.truth._target_.endswith("CocoaTruth")
    assert list(c.plots.draw) == list(c.task.plots)
    cat = catalogue.load(c.dataset.module)
    assert {"jet_e", "n_tracks"} <= set(cat.observables)
    assert {"jet_e_response", "jet_dr"} <= set(cat.comparisons)
    assert {"jet_e", "response_vs_pt"} <= set(cat.plots)
    s = cfg("task=pflow", "dataset=synthetic")
    assert s.dataset.truth._target_.endswith("ContractTruth")
    assert s.dataset.module == c.dataset.module


def test_metric_and_param_names_are_checked():
    c = cfg("task=pflow", "task.metrics.1.x=jet_pt")
    cat = bench_steps.dataset_catalogue(c)
    with pytest.raises(ValueError, match="truth.jet_pt or reco.jet_pt"):
        bench_steps.task_metrics(c, cat)
    with pytest.raises(ValueError, match=r"task.params \['match_r'\]"):
        bench_steps.check_params(cfg("task=pflow", "+task.params.match_r=0.1"), cat)
