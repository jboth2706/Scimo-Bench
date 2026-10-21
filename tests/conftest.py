import os
import subprocess
import sys
from pathlib import Path

import pytest

from scimo_bench.io import REPO_ROOT
from scimo_bench.jobs import ENV_ALLOW, ENV_JOB
from tests import synthetic

# the suite is small and runs where it is called, login nodes included (`pixi run test-job`
# runs it as a job)
os.environ.setdefault(ENV_ALLOW, "1")

JOB_VARS = (ENV_JOB, "SLURM_JOB_ID", ENV_ALLOW)
MODELS_FILE = "models_file=tests/models.py"

COCOA_VAL = Path(
    "/ptmp/jboth/COCOA/COCOA_samples_singlejet/val_100K_cells256/singlejet_10_100kseg_bw100.0.root"
)


@pytest.fixture(scope="session")
def truth():
    return synthetic.make_truth(400, seed=0)


@pytest.fixture(scope="session")
def contract_dir(tmp_path_factory, truth):
    """Truth + three synthetic models written as contract files."""
    d = tmp_path_factory.mktemp("contract")
    synthetic.write_truth(d, truth)
    synthetic.write(d, "ident", synthetic.identical(truth))
    synthetic.write(d, "smear", synthetic.smeared(truth, 0.1), dict(mode="smear", sigma=0.1))
    synthetic.write(d, "noe", synthetic.missing_field(synthetic.shifted(truth, 1.1)))
    return d


def run_bench(*overrides, check=True, env=None):
    """Run `python -m scimo_bench` from the repo root; returns the CompletedProcess."""
    proc = subprocess.run(
        [sys.executable, "-m", "scimo_bench", *overrides],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    if check and proc.returncode != 0:
        raise AssertionError(f"scimo_bench failed:\n{proc.stdout}\n{proc.stderr}")
    return proc


def synthetic_overrides(contract_dir, out_dir, models=("ident", "smear")):
    """Overrides that point the test models `dummy`/`second` at synthetic files."""
    names = ["dummy", "second"][: len(models)]
    ov = [
        MODELS_FILE,
        "task=pflow",
        f"models=[{','.join(names)}]",
        "dataset=synthetic",
        f"dataset.truth.path={contract_dir / 'truth'}",
        "task.events=null",
        f"output.dir={out_dir}",
        "output.bootstrap.n=50",
        "output.formats=[png,pdf]",
    ]
    for n, m in zip(names, models, strict=True):
        ov.append(f"models.{n}.predictions={contract_dir / m}")
    return ov


@pytest.fixture
def fake_slurm(tmp_path):
    """`sbatch` that records its arguments and prints a job id; `srun` that just runs."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "sbatch").write_text(
        f'#!/bin/bash\necho "$@" >> {tmp_path / "sbatch.log"}\necho "12345;raven"\n'
    )
    (bin_dir / "srun").write_text('#!/bin/bash\nexec "$@"\n')
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in JOB_VARS}
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    return env
