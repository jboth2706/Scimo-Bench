"""Integration on 1,000 real COCOA val events (quality path, CPU)."""

import numpy as np
import pandas as pd
import pytest

from scimo_bench import io
from scimo_bench.datasets.cocoa.reader import CocoaTruth
from tests import synthetic
from tests.conftest import COCOA_VAL, MODELS_FILE, run_bench

pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(not COCOA_VAL.exists(), reason="COCOA val file not available"),
]


def test_real_events_end_to_end(tmp_path):
    ids = io.read_event_list("data/pflow_val_v1_events.txt")[:1000]
    events = tmp_path / "events.txt"
    np.savetxt(events, ids, fmt="%d")
    truth = CocoaTruth(COCOA_VAL).read(ids)
    synthetic.write(tmp_path, "ident", synthetic.identical(truth))
    synthetic.write(tmp_path, "shift", synthetic.shifted(truth, 1.1))
    out = tmp_path / "run"
    run_bench(
        MODELS_FILE,
        "task=pflow",
        "models=[dummy,second]",
        f"dataset.truth.path={COCOA_VAL}",
        f"models.dummy.predictions={tmp_path / 'ident'}",
        f"models.second.predictions={tmp_path / 'shift'}",
        f"task.events={events}",
        "cutflow=preselection",
        f"output.dir={out}",
        "output.bootstrap.n=100",
    )
    rep = pd.read_csv(out / "report.csv")
    assert set(rep.status) <= {"ok", "invalid"}
    ov = rep[rep.bin_low.isna()].set_index(["model", "metric"])["value"]
    assert ov["dummy", "response_median"] == pytest.approx(1.0)
    assert ov["dummy", "response_iqr"] == pytest.approx(0.0, abs=1e-6)
    # the 1.1x pT cut at 1 GeV moves a few particles across it; the response stays ~1.1
    assert ov["second", "response_median"] == pytest.approx(1.1, abs=0.01)
    cf = pd.read_csv(out / "cutflow.csv")
    join = cf[cf.step == "join"].set_index("model")
    assert join.loc["truth", "events"] == 1000 and join.loc["dummy", "missing_events"] == 0
    assert (out / "plots" / "response_vs_pt.pdf").exists()
