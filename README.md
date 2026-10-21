# SciMo-Bench

SciMo-Bench benchmarks scientific foundation models. It compares the predictions of several
models with the truth (same events, same cuts, same binning, intervals from a paired
bootstrap), measures what each model costs to run, and combines both on Pareto fronts.

- **Quality**: physics metrics per model, per bin, with 95 % intervals; rankings; plots.
- **Cost**: latency, throughput, GPU-seconds, memory, FLOPs, measured honestly (warmup and
  compile separate, CUDA-synchronized), counted only if the model's quality passes a gate.
- **Any model family**: a model is handed in either as a small Python class that
  SciMo-Bench runs (`docs/PROTOCOL.md`), or as a prediction file (`docs/CONTRACT.md`).

Configs are pure Python ([hydra-zen](https://github.com/mit-ll-responsible-ai/hydra-zen));
everything is set on the command line. On a cluster, each call is submitted as small CPU /
GPU jobs; nothing heavy runs on the login node. "SciMo-Bench" is a working name.

---

## Quick start

**1. Install** ([pixi](https://pixi.sh)):

```bash
git clone git@github.com:jboth2706/Scimo-Bench.git scimo-bench && cd scimo-bench
pixi install                                    # scoring environment (CPU, no torch)
CONDA_OVERRIDE_CUDA=12.9 pixi install -e perf   # only to run models: torch + model code
```

The `perf` environment installs the model packages as editable sibling checkouts
(`../scimo`, `../scimo-framework`); see `pixi.toml`.

**2. List your models** in a small Python file of your own (not in this repo; example:
`examples/hep4m_models.py`):

```python
# ~/bench_models.py
from scimo_bench.configs.models import add_model

# run by SciMo-Bench through its protocol class
add_model("n6_muon", type="nanohep", label="nanoHEP n6 Muon",
          checkpoint="/ptmp/me/runs/<run>/checkpoints/final.ckpt")
# prediction files made elsewhere; only scored
add_model("hgpflow", family="regression", predictions="/ptmp/me/bench/preds/hgpflow")
```

**3. Run.** From anywhere (`run.sh` lives in the repo):

```bash
# run the models, time them, score them, plot (on Raven: a GPU job, then a CPU job)
./run.sh site=raven models_file=~/bench_models.py task=pflow \
    'models=[n6_muon,hgpflow]' steps=[predict,perf,score,plot] perf=single cutflow=central

# only score and plot prediction files that exist (the default steps)
./run.sh site=raven models_file=~/bench_models.py task=pflow 'models=[n6_muon,hgpflow]'

# redraw, with quality-vs-cost plots (runs right where you call it)
./run.sh site=raven models_file=~/bench_models.py task=pflow 'models=[n6_muon,hgpflow]' \
    steps=[plot] plots=pareto output.dir=<run folder>
```

**4. Look at the results.** The call prints the run folder
(`<paths.outputs>/<task>/<time>/`):

```bash
squeue --me                                   # the jobs
tail -f <run folder>/job-*.out                # their logs
column -s, -t < <run folder>/rankings.csv     # ranks per metric
ls <run folder>/plots                         # PNG + PDF
```

---

## How it works

### Two sides, one boundary

| Your model repo | SciMo-Bench |
|-----------------|-------------|
| model code, training, checkpoints | the task: events, cuts, metrics, plots |
| the model's own input data (e.g. tokenized events) | runs your class over those events, times it |
| one class turning the model's output (tokens, bins, normalized values) into floats | compares with the truth; reports, rankings, plots, Pareto fronts |
| | submits the work as cluster jobs |

Only floats per event cross the boundary: `event_id` plus each object's fields in physical
units (for COCOA: `particles.pt, eta, phi, e` in GeV and rad). SciMo-Bench never sees tokens,
vocabularies or checkpoints.

### Steps

A call runs some of four steps, always in this order (`steps=[...]`, default
`[score,plot]`):

```
                     model code (GPU job on a cluster)          files only (CPU job)
 your class ──> predict ──> prediction files ───────┐
          └───> perf ─────> perf.csv ───────────────┤
                                                    v
 truth (dataset reader) ──> io: read, dedupe, join on event_id
                            └> cutflow ─> variables ─> scorer ─> report.csv, rankings.csv, ...
                                                                 + perf rows after the gate
                                                                   └> plotter ─> plots/, pareto.csv
```

| Step | Does | Writes |
|------|------|--------|
| `predict` | loads each model with a class once, runs it over the task's events | `<paths.preds>/<task>/<model>/` (Parquet + `meta.json`) |
| `perf` | times the same loaded model (scenarios `single`, `offline`) | `perf.csv`, `perf_meta_<perf>.json` |
| `score` | reads truth and prediction files, cuts, computes variables and metrics | `report.csv`, `rankings.csv`, `cutflow.csv`, `per_event.parquet`, `models.csv` |
| `plot` | draws from those files (no physics) | `plots/`, `pareto.csv` |

Timings are counted only if the same model and generation settings pass the task's quality
gate (`pflow`: jet-energy-response IQR ≤ 0.25); otherwise they are marked `invalid`.

### Where it runs

| `site=` | Behaviour |
|---------|-----------|
| `local` (default) | everything in-process (laptop, or inside an `salloc` shell) |
| `raven` | `predict`/`perf` as one job on the runtime's device (1× A100), then `score`/`plot` as a CPU job that starts only if the first succeeded; the call returns at once (`site.wait=true` blocks) |

Before submitting, the call fixes the run folder and copies the code and your models file
to a snapshot, so later edits do not change queued jobs. Job scripts, Slurm logs and job ids
(`jobs.csv`) land in the run folder. On a login node of a known cluster, every step except
`plot` is refused unless you pass the site ("add site=raven").

---

## Using it

### What you set per call

| Setting | Meaning | Default |
|---------|---------|---------|
| `site=` | `local` or `raven` | `local` |
| `paths.data` | data root; the dataset's truth file is found below it | raven: `/ptmp/$USER`; local: none |
| `paths.preds` | prediction files, `<preds>/<task>/<model>/` | raven: `/ptmp/$USER/bench/preds`; local: `outputs/preds` |
| `paths.outputs` | run folders, `<outputs>/<task>/<time>/` | raven: `/ptmp/$USER/bench/outputs`; local: `outputs` |
| `paths.snapshots` | code copies for jobs | raven: `/ptmp/$USER/bench/snapshots` |
| `models_file=` | your file of `add_model(...)` lines | none |
| `models=[a,b]` | which of your models | none |
| `steps=[...]` | `predict`, `perf`, `score`, `plot` | `[score,plot]` |
| `task=` | what is scored: `pflow` | required |
| `dataset=` | eval dataset: `cocoa` | the task's |
| `runtime=` | how models run: `gpu` (cuda, bf16, compiled, batch 256) or `cpu` | `gpu` |
| `perf=` | `single` (latency), `offline` (throughput), `both`, `smoke` (code check) | none |
| `cutflow=` | cuts applied to every model: `central` | none |
| `plots=` | `default`, or `pareto` (adds quality vs cost) | `default` |
| `output.dir=` | an existing run folder (for `steps=[plot]` / `[perf]`) | new folder; latest run for plot/perf |

Any value can be overridden, e.g. `paths.data=/elsewhere`,
`models.n6_muon.checkpoint=<file>`, `output.bootstrap.n=200`. The settings used are saved as
`resolved_config.yaml` in the run folder. Relative paths are relative to where you call
`./run.sh` (`models_file`) or to this repo (task files such as event lists).

### Models

`add_model(name, type=..., **fields)` with fields `label, family, predictions, generation,
model, checkpoint, options, env, color, linestyle` (`scimo_bench/configs/models.py`); a
typo is an error. A type gives defaults: `files` (only a prediction file) or `nanohep`
(protocol class `hep4m_scimo.bench:NanoHEP`, argmax generation, env `perf`); `add_type(...)`
adds one.

### The run folder

| File | Holds |
|------|-------|
| `report.csv` | every metric per model and bin: value, 95 % interval, status; gated perf rows |
| `rankings.csv` | rank per metric, mean rank, primary metric |
| `cutflow.csv` | events and objects kept after each step, per model |
| `per_event.parquet` | variable values per event (for plots and later analysis) |
| `perf.csv` | timings as measured, before the gate |
| `pareto.csv` | quality-vs-cost points, on the front or not |
| `plots/` | PNG and PDF |
| `run.log`, `jobs.csv`, `job-*.sh`, `job-*.out` | commit, warnings, jobs |
| `resolved_config*.yaml` | the full configuration used |

**Statuses**: `ok`; `missing` (a model without predictions, a variable lacking an input
field; warning, the run goes on); `invalid` (empty bin, or a failed quality gate).
`output.on_missing=error` turns warnings into errors. A model whose class fails is logged,
and its old prediction file is removed first, so it is never scored stale.

---

## Making a model runnable

Write one class in your repo (it needs nothing from SciMo-Bench); full description in
[`docs/PROTOCOL.md`](docs/PROTOCOL.md):

```python
class MyModel:
    def load(self, cfg): ...               # once: checkpoint, generation, options, runtime
    def inputs(self, event_ids, size): ... # the model's own inputs for these events, on device
    def generate(self, batch): ...         # one batch through the model (this is timed)
    def to_floats(self, out): ...          # event_id + objects as floats (GeV, rad)
    def info(self): ...                    # family, n_params, generation, flops_per_event, ...
```

Register it with `add_model("my_model", model="my_pkg.bench:MyModel", checkpoint=...,
env="perf")`. Without a class, write a prediction file in the format of
[`docs/CONTRACT.md`](docs/CONTRACT.md) and point `predictions=` at it.

---

## Datasets, variables and plots

### Datasets

What can be computed and drawn depends on the data. An eval dataset bundles it; a task names
its default (`pflow` → `cocoa`), and `dataset=<name>` swaps all of it at once.

| Part | COCOA (`dataset=cocoa`) | Where |
|------|-------------------------|-------|
| truth reader | raw ROOT, tree `EventTree`, `particle_*` branches → `particles.{pt,eta,phi,e,pdgid}` | `scimo_bench/datasets/cocoa.py` |
| variables | jets (vector sum), per particle, ΔR matching, event EMD, plus the generic ones | `datasets/cocoa.py`, `variables.py` |
| what predictions must contain | `particles: [pt, eta, phi, e]` | `configs/datasets.py` |
| plot catalogue | every plot that makes sense for it, loaded into `plots.specs` | `configs/datasets.py` |

A new dataset: a module in `scimo_bench/datasets/` with a reader class
(`read(event_ids) -> ak.Array`) and a `REGISTRY` of variables, then one `add_dataset(...)`
call in `scimo_bench/configs/datasets.py`.

### Variables

Every quantity is a short function in the dataset's module, registered by name:

```python
@REGISTRY.variable("jet_mass", label="jet mass", unit="GeV")
def _jet_mass(truth, pred, ctx):
    """Invariant mass of the predicted jet."""
    return ...  # one value per event (numpy); level="object" for one per object (awkward)
```

Options: `level="object"`, `inputs="truth"` (truth only), `reference=` (the ideal value,
drawn as a line), `truth=` (the truth counterpart, overlaid in plots). Helpers (`obj_field`,
`match`, `emd_event`, ...) are in `scimo_bench/variables.py`; task parameters come from
`ctx.param(...)`. A registered variable can be scored by a metric, stored
(`task.variables`) and plotted by name: `plots.extra=[jet_mass]` draws its distribution.

### Plots

Every option of every plot is a config field (`scimo_bench/plotter.py`: `Distribution`,
`MetricVsX`, `RankingPlot`, `ParetoPlot`):

```bash
plots.specs.jet_pred_e.bins='${linspace:0,300,61}'  plots.specs.jet_pred_e.log_y=true
plots.specs.jet_pred_e.ratio_range=[0.8,1.2]          plots.specs.jet_pred_e.title='jet E'
'+plots.specs.jet_eta={_target_: scimo_bench.plotter.Distribution, variable: jet_pred_eta, bins: 30}'
plots.extra=[jet_eta]  plots.skip=[ranking]           # or plots.draw=[...] to replace the list
models.n6_muon.color=red  models.n6_muon.linestyle=--  plots.style.font_size=12
```

`plots.style` sets the look of all plots (palette, line styles, sizes, fonts, any matplotlib
setting under `rc`). Colours are matplotlib names (`red`, `navy`, `tab:orange`). A new plot
kind: a class with `name` and `draw(pc) -> list[Path]`, decorated `@register_plot("kind")`.

---

## Repository structure

```
scimo-bench/
├── run.sh                    entry point: ./run.sh <hydra overrides>
├── pixi.toml, pixi.lock      environments: default (scoring), perf (torch + model code)
├── scimo_bench/
│   ├── main.py               loads the models file, composes the config, runs or submits
│   ├── steps.py              the steps predict/perf, score, plot (each works on files)
│   ├── configs/              hydra-zen config groups (below)
│   ├── datasets/             eval datasets: truth readers + variable registries (cocoa.py)
│   ├── protocol.py           the model protocol (the only interface to model code)
│   ├── predict.py            step predict: run a model, write its prediction file
│   ├── perf.py               runtime, timing scenarios, quality gate
│   ├── schema.py, io.py      prediction file format (contract v1), reading, join on event_id
│   ├── variables.py          variable registry, helpers, generic variables
│   ├── cutflow.py            cuts, built on variables
│   ├── scorer.py             metrics, paired bootstrap, statuses, rankings
│   ├── plotter.py            plots, style, Pareto fronts (no physics)
│   └── jobs.py               login-node guard, job scripts, submission, code snapshots
├── data/                     event lists of the tasks (versioned with the task)
├── examples/                 example models file
├── docs/                     PLAN, DESIGN, PROTOCOL, CONTRACT, task briefs per milestone
└── tests/                    pytest: synthetic data with known answers, a toy model
```

### Config groups (`scimo_bench/configs/`)

| Group | Chooses | Options |
|-------|---------|---------|
| `site` | where it runs, default paths, job resources | `local`, `raven` |
| `task` | events, metric definitions, binning, default plots, quality gate | `pflow` |
| `dataset` (+ `plot_specs`) | truth reader, variables, required fields, plot catalogue | `cocoa` |
| `models` | from your models file | yours |
| `runtime` | device, precision, compile, batch size for model code | `gpu`, `cpu` |
| `perf` | scenarios, batch sizes, warmup, repeats | `single`, `offline`, `both`, `smoke` |
| `cutflow` | cuts applied to every model | `central` |
| `plots` (+ `plots/pareto`) | selection, style, Pareto plot | `default`, `pareto` |
| `output` | run folder, bootstrap size, warn or stop on gaps | `default` |

A new cluster is one `site_conf(...)` entry in `configs/sites.py` (Slurm); a new task is
one entry in `configs/tasks.py`.

### Design rules

The quality path never runs models; model code runs only through the protocol, in jobs.
Every physics quantity is one registered variable; cuts, scorer and plotter compute none.
All models are scored on the same events, cuts and binning. A gap for one model never
aborts the run. Results are files, and plots can be redrawn from them. Details in
[`docs/DESIGN.md`](docs/DESIGN.md) and `CLAUDE.md`.

---

## Development

```bash
pixi run test            # all tests, in-process (~2 min); real-data tests skip without /ptmp
pixi run test-job raven  # the same tests as a CPU job
pixi run lint            # ruff check + format check
pixi run fmt             # format
```

`run.sh` and the test task set `OPENBLAS_NUM_THREADS=1`: on a shared node an OpenBLAS pool
sized for all cores inside a small cgroup made scoring ~150× slower.

## Status

| # | Milestone | State |
|---|-----------|-------|
| M0 | scaffold, hydra-zen store, output folder | done |
| M1 | contract v1, IO, join, synthetic data | done; COCOA val: 99,955 events, 413 duplicate particle pairs (408 objects dropped) |
| M2 | nanoHEP class in hep4m_scimo implementing the protocol (incl. tokens → floats) | open |
| M3 | variables, cutflow | done |
| M4 | scorer, rankings, report files | done |
| M5 | plotter, redraw from files | done; redraw is byte-identical |
| M6 | perf path, model protocol, predict step | done; tested on CPU and on Raven with a toy model; GPU measurement needs M2 |
| M7 | Pareto front | done; real points need M2 |

Decisions taken while building (proposals in [`docs/PLAN.md`](docs/PLAN.md)): event list =
first 20,000 `eventNumber`s of the val file; greedy ΔR < 0.05 matching by truth pT;
duplicate truth objects dropped; multiplicity from the model, oracle results labelled in
plots; the overall value of a binned metric uses all events.
