import numpy as np
import pandas as pd
import pytest

from scimo_bench import perf, protocol
from scimo_bench.perf import PerfSettings, Runtime, apply_gate, gate_status, merge_rows, run_model
from scimo_bench.schema import condition_hash
from scimo_bench.scorer import REPORT_COLUMNS

GEN = dict(mode="argmax", multiplicity="model")
SETTINGS = PerfSettings(
    name="cpu",
    scenarios=["single", "offline"],
    batch_sizes=[1, 8],
    warmup=2,
    repeats=3,
    n_events=64,
    single_events=16,
)
RT = Runtime(name="cpu", device="cpu", precision="fp32", compile=False, batch_size=64)
EVENTS = np.arange(1000, 1200)


def model_cfg(model="tests.toy_model:ToyModel", name="toy"):
    return dict(name=name, generation=GEN, model=model)


def quality(value, status="ok", gen=GEN, model="toy"):
    return pd.DataFrame(
        [
            dict(
                model=model,
                metric="response_iqr",
                bin_low=np.nan,
                bin_high=np.nan,
                value=value,
                lower=value,
                upper=value,
                status=status,
                kind="quality",
                task_version=1,
                condition=f"gen={condition_hash(gen)}",
            )
        ],
        columns=REPORT_COLUMNS,
    )


GATE = dict(metric="response_iqr", max=0.25)


def test_toy_model_rows():
    rows, meta = run_model(model_cfg(), SETTINGS, RT, EVENTS, 1)
    df = pd.DataFrame(rows)
    unreported = ["peak_memory", "nfe_per_event", "tokens_per_event"]  # CPU / not in info()
    assert set(df.kind) == {"perf"}
    assert set(df[~df.metric.isin(unreported)].status) == {"ok"}
    assert set(df[df.metric.isin(unreported)].status) == {"missing"}
    conds = sorted(set(df.condition))
    assert len(conds) == 3  # single bs1, offline bs1, offline bs8
    assert all(c.startswith(f"gen={condition_hash(GEN)};scenario=") for c in conds)
    single = df[df.condition.str.contains("scenario=single")].set_index("metric")
    assert {"latency_p50", "latency_p90", "latency_p99", "latency_e2e_p50"} <= set(single.index)
    assert single.loc["latency_p50", "value"] <= single.loc["latency_p99", "value"]
    assert (
        single.loc["latency_p50", "lower"]
        <= single.loc["latency_p50", "value"]
        <= single.loc["latency_p50", "upper"]
    )
    # compile and warmup are rows of their own
    assert {"compile_s", "warmup_s", "load_s"} <= set(single.index)
    off = df[df.condition.str.contains("scenario=offline;batch_size=8")].set_index("metric")
    assert off.loc["throughput", "value"] > 0
    assert off.loc["gpu_s_per_1k", "value"] == pytest.approx(
        1000 / off.loc["throughput", "value"], rel=0.3
    )
    assert off.loc["peak_memory", "status"] == "missing"  # CPU: no CUDA memory stats
    assert off.loc["n_params", "value"] == 32 * 32
    assert off.loc["flops_per_event", "value"] == 2 * 32 * 32 * 8
    assert meta["hardware"]["device"] == "cpu" and meta["model"]["n_params"] == 1024


def test_gate_is_applied_after_measuring():
    rows = pd.DataFrame(run_model(model_cfg(), SETTINGS, RT, EVENTS, 1)[0])
    finite = rows.value.notna()
    assert set(apply_gate(rows, quality(0.1), GATE)[finite].status) == {"ok"}
    assert set(apply_gate(rows, quality(0.5), GATE)[finite].status) == {"invalid"}
    # no quality run yet; unmeasured rows stay missing either way
    assert set(apply_gate(rows, None, GATE).status) == {"missing"}
    assert set(apply_gate(rows, quality(0.5), GATE)[~finite].status) == {"missing"}


def test_gate_needs_a_quality_row_of_the_same_generation():
    # a quality row of another generation config does not count
    q = quality(0.1, gen=dict(mode="sample", temperature=1.0))
    assert gate_status(q, "toy", condition_hash(GEN), GATE) == "missing"
    assert (
        gate_status(quality(0.1, status="invalid"), "toy", condition_hash(GEN), GATE) == "invalid"
    )
    assert gate_status(quality(0.1), "toy", condition_hash(GEN), None) == "ok"
    assert (
        gate_status(quality(0.1), "toy", condition_hash(GEN), dict(metric="response_iqr", min=0.2))
        == "invalid"
    )


def test_model_without_class():
    rows, meta = run_model(model_cfg(model=None), SETTINGS, RT, EVENTS, 1)
    assert rows and {r["status"] for r in rows} == {"missing"}
    assert meta["model"] is None


def test_static_info_missing_when_not_reported():
    rows, _ = run_model(model_cfg("tests.toy_model:NoInfoModel"), SETTINGS, RT, EVENTS, 1)
    df = pd.DataFrame(rows)
    assert (df[df.metric == "n_params"].status == "missing").all()
    assert (df[df.metric == "throughput"].status == "ok").all()


def test_merge_rows_replaces_same_condition():
    rows, _ = run_model(model_cfg(), SETTINGS, RT, EVENTS, 1)
    rep = merge_rows(quality(0.1), rows)
    again = merge_rows(rep, rows)
    assert len(again) == len(rep) == 1 + len(rows)
    assert list(again.columns) == REPORT_COLUMNS


def test_settings_validation():
    with pytest.raises(ValueError, match="scenarios"):
        PerfSettings.from_cfg(dict(name="x", scenarios=["rate"], batch_sizes=[1]))
    if perf.torch is None:
        with pytest.raises(RuntimeError, match="CUDA"):
            Runtime.from_cfg(dict(name="gpu", device="cuda"))


def test_bad_model_spec():
    with pytest.raises(ValueError, match="package.module:Class"):
        protocol.create("tests.toy_model.ToyModel")
    with pytest.raises(TypeError, match="missing"):
        protocol.create("tests.toy_model:Incomplete")
