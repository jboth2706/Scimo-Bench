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
add_model(
    "n6_muon",
    type="nanohep",
    label="nanoHEP n6 Muon",
    checkpoint="/ptmp/me/runs/<run>/checkpoints/final.ckpt",
)
# prediction files made elsewhere; only scored
add_model("hgpflow", family="regression", predictions="/ptmp/me/hgpflow_preds")
```

**3. Run.** From anywhere (`run.sh` lives in the repo). Every call names its run
(`run=<name>`); everything of the run, predictions included, goes into
`<paths.outputs>/<name>/` (on Raven `/ptmp/$USER/bench/outputs/<name>/`):

```bash
# run the models, time them, score them, plot (on Raven: a GPU job, then a CPU job)
./run.sh site=raven run=n6_vs_hgpflow models_file=~/bench_models.py task=pflow \
    'models=[n6_muon,hgpflow]' steps=[predict,perf,score,plot] perf=single cutflow=preselection

# score the same run again (e.g. after adding a variable): the predictions are kept
./run.sh site=raven run=n6_vs_hgpflow models_file=~/bench_models.py task=pflow \
    'models=[n6_muon,hgpflow]' cutflow=preselection --force

# redraw, with quality-vs-cost plots (runs right where you call it)
./run.sh site=raven run=n6_vs_hgpflow models_file=~/bench_models.py task=pflow \
    'models=[n6_muon,hgpflow]' steps=[plot] plots=pareto
```

A call with `predict` or `score` into a run that exists is refused; `--force` empties the run
folder first but keeps its `predictions/`. `steps=[plot]` and `steps=[perf]` work on an
existing run.

**4. Look at the results**:

```bash
squeue --me                                                    # the jobs
tail -f /ptmp/$USER/bench/outputs/n6_vs_hgpflow/job-*.out      # their logs
column -s, -t < /ptmp/$USER/bench/outputs/n6_vs_hgpflow/rankings.csv
ls /ptmp/$USER/bench/outputs/n6_vs_hgpflow/plots               # PNG + PDF
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
                            └> cutflow ─> observables, comparisons ─> scorer ─> report.csv, ...
                                                                 + perf rows after the gate
                                                                   └> plotter ─> plots/, pareto.csv
```

| Step | Does | Writes |
|------|------|--------|
| `predict` | loads each model with a class once, runs it over the task's events | `<run>/predictions/<model>/` (Parquet + `meta.json`) |
| `perf` | times the same loaded model (scenarios `single`, `offline`) | `perf.csv`, `perf_meta_<perf>.json` |
| `score` | reads truth and prediction files, cuts, computes observables, comparisons and metrics | `report.csv`, `rankings.csv`, `cutflow.csv`, `per_event.parquet`, `models.csv` |
| `plot` | draws from those files (no physics) | `plots/`, `pareto.csv` |

Timings are counted only if the same model and generation settings pass the task's quality
gate (`pflow`: jet-energy-response IQR ≤ 0.25); otherwise they are marked `invalid`.

### Where it runs

| `site=` | Behaviour |
|---------|-----------|
| `local` (default) | everything in-process (laptop, or inside an `salloc` shell) |
| `raven` | `predict`/`perf` as one job on the runtime's device (1× A100), then `score`/`plot` as a CPU job that starts only if the first succeeded; the call returns at once (`site.wait=true` blocks) |

Before submitting, the call prepares the run folder and copies the code and your models file
to a snapshot, so later edits do not change queued jobs. Job scripts, Slurm logs and job ids
(`jobs.csv`) land in the run folder. On a login node of a known cluster, every step except
`plot` is refused unless you pass the site ("add site=raven").

---

## Using it

### What you set per call

| Setting | Meaning | Default |
|---------|---------|---------|
| `run=` | the run's name: its folder is `<paths.outputs>/<name>/` | required |
| `--force` | overwrite an existing run (its `predictions/` are kept) | off |
| `site=` | `local` or `raven` | `local` |
| `paths.data` | data root; the dataset's truth file is found below it | raven: `/ptmp/$USER`; local: none |
| `paths.outputs` | where the run folders are | raven: `/ptmp/$USER/bench/outputs`; local: `outputs` |
| `paths.snapshots` | code copies for jobs | raven: `/ptmp/$USER/bench/snapshots` |
| `models_file=` | your file of `add_model(...)` lines | none |
| `models=[a,b]` | which of your models | none |
| `steps=[...]` | `predict`, `perf`, `score`, `plot` | `[score,plot]` |
| `task=` | what is scored: `pflow` | required |
| `dataset=` | eval dataset: `cocoa` | the task's |
| `runtime=` | how models run: `gpu` (cuda, bf16, compiled, batch 256) or `cpu` | `gpu` |
| `perf=` | `single` (latency), `offline` (throughput), `both`, `smoke` (code check) | none |
| `cutflow=` | cuts applied to every model, each on truth, reco or both: `preselection` | none |
| `plots=` | `default`, or `pareto` (adds quality vs cost) | `default` |

Any value can be overridden, e.g. `paths.data=/elsewhere`,
`models.n6_muon.checkpoint=<file>`, `output.bootstrap.n=200`. The settings used are saved as
`resolved_config.yaml` in the run folder. Relative paths are relative to where you call
`./run.sh` (`models_file`) or to this repo (task files such as event lists).

### Models

`add_model(name, type=..., **fields)` with fields `label, family, predictions, generation,
model, checkpoint, options, env, color, linestyle` (`scimo_bench/configs/models.py`); a
typo is an error. `predictions` defaults to `<run>/predictions/<model>`; point it at another
run's folder (`predictions=/ptmp/$USER/bench/outputs/<other run>/predictions/<model>`) to
score predictions made before without rerunning the model. A type gives defaults: `files` (only a prediction file) or `nanohep`
(protocol class `examples.nanohep_model:NanoHEP`, argmax generation, env `perf`); `add_type(...)`
adds one.

### The run folder

`<paths.outputs>/<run>/`:

| File | Holds |
|------|-------|
| `predictions/<model>/` | each model's predictions (Parquet + `meta.json`, `docs/CONTRACT.md`) |
| `report.csv` | every metric per model and bin: value, 95 % interval, status; gated perf rows |
| `rankings.csv` | rank per metric, mean rank, primary metric |
| `cutflow.csv` | events and objects kept after each step, per model |
| `per_event.parquet` | every observable (truth and reco) and comparison per event, for plots and later analysis |
| `perf.csv` | timings as measured, before the gate |
| `pareto.csv` | quality-vs-cost points, on the front or not |
| `plots/` | PNG and PDF |
| `run.log`, `jobs.csv`, `job-*.sh`, `job-*.out` | commit, warnings, jobs |
| `resolved_config*.yaml` | the full configuration used |

**Statuses**: `ok`; `missing` (a model without predictions, a quantity lacking an input
field; warning, the run goes on); `invalid` (empty bin, or a failed quality gate).
`output.on_missing=error` turns warnings into errors. A model whose class fails is logged,
and its old prediction file is removed first, so it is never scored stale.

---

## Making a model runnable

Write one class in your repo (it needs nothing from SciMo-Bench); full description in
[`docs/PROTOCOL.md`](docs/PROTOCOL.md):

```python
class MyModel:
    def load(self, cfg): ...  # once: checkpoint, generation, options, runtime
    def inputs(self, event_ids, size): ...  # the model's own inputs for these events, on device
    def generate(self, batch): ...  # one batch through the model (this is timed)
    def to_floats(self, out): ...  # event_id + objects as floats (GeV, rad)
    def info(self): ...  # family, n_params, generation, flops_per_event, ...
```

Register it with `add_model("my_model", model="my_pkg.bench:MyModel", checkpoint=...,
env="perf")`. Without a class, write a prediction file in the format of
[`docs/CONTRACT.md`](docs/CONTRACT.md) and point `predictions=` at it.

---

## Datasets

What can be computed and drawn depends on the data. An eval dataset bundles it; a task names
its default (`pflow` → `cocoa`), and `dataset=<name>` swaps all of it at once. Each dataset
is one package with four files, each with one job:

```
scimo_bench/datasets/cocoa/
    reader.py        the truth file → events (collections `particles`, `tracks`)
    observables.py   quantities of ONE set of events: jet pT, particles per event, ...
    comparisons.py   quantities of the truth AND one model: responses, matching, EMD
    plots.py         one function per plot
scimo_bench/datasets/common.py   physics helpers any dataset can use (jet, ΔR, matching, EMD)
```

| COCOA (`dataset=cocoa`) | |
|-------------------------|---|
| truth | raw ROOT, tree `EventTree`: `particles.{pt,eta,phi,e,pdgid}` and `tracks.{pt,eta,phi,d0,z0,particle_idx}` |
| models must output | `particles: [pt, eta, phi, e]` (`configs/datasets.py`) |
| tracks | an input of the models, so they exist in the truth only |

A new dataset: a package like `cocoa/` (its `__init__.py` has
`CATALOGUE = Catalogue.collect(observables, comparisons, plots)`) and one `add_dataset(...)`
call in `scimo_bench/configs/datasets.py`.

---

## Adding variables

There are two kinds of quantity, and both are plain functions; the function name is the
name everywhere (configs, `per_event.parquet`, plots).

**Observables** (`observables.py`) are quantities of one set of events. SciMo-Bench runs each
one on the truth **and** on every model's output (reco), so you write it once:

```python
@observable(label="jet mass", unit="GeV")
def jet_mass(ev):
    pt, eta, phi = f64(ev.particles.pt), f64(ev.particles.eta), f64(ev.particles.phi)
    e, px, py, pz = pt * np.cosh(eta), pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    s = lambda x: ak.to_numpy(ak.sum(x, axis=1))
    return np.sqrt(np.maximum(s(e) ** 2 - s(px) ** 2 - s(py) ** 2 - s(pz) ** 2, 0))


@observable(label="tracks per event")
def n_tracks(ev):
    return counts(ev.tracks.pt).astype(np.float64)
```

- `ev` holds the events after the join and the cuts: `ev.particles.pt`, `ev.tracks.d0`, ...
- Return one value per event (numpy array) or a list per event (awkward array), e.g.
  `ev.particles.pt` for one value per particle.
- An observable is computed on every side that has its inputs. If a model's output lacks a
  field the dataset requires (e.g. `particles.e`), that model gets `missing` for it, with a
  warning. A collection models are not asked for (`tracks`) is silently truth only.

**Comparisons** (`comparisons.py`) need the truth and one model, row by row:

```python
@comparison(label=r"$m_\mathrm{reco} / m_\mathrm{truth}$ (jet)", reference=1.0)
def jet_mass_response(truth, reco):
    return ratio(jet_mass(reco), jet_mass(truth))


@comparison(label="truth particle matched to reco", reference=1.0)
def particle_matched(truth, reco, match_dr=0.05):
    m = match(truth, reco, match_dr)
    return unflatten((m["truth_idx"] >= 0).astype(np.float64), m["truth_counts"])
```

- A comparison may call observables (`jet_mass(reco)`) and the helpers in
  `datasets/common.py` (`jet`, `match`, `emd`, `delta_phi`, `ratio`, `counts`, `flat`, ...).
- Task parameters are keyword arguments with a default; the task's `params` override them
  (`task.params.match_dr=0.1`). A parameter no comparison has is a configuration error.
- `reference` is the ideal value (drawn as a dotted line). Let the label say which side
  each part comes from (reco, truth).

Every observable and comparison is computed in every `score` step and stored in
`per_event.parquet` (one row per event: `model` = `truth` or the model name). Then you can

- **plot** it: write a plot function (next section);
- **score** it: add a metric to the task in `configs/tasks.py`, naming a comparison, or an
  observable with its side:
  ```python
  builds(
      Median,
      name="jet_mass_response",
      variable="jet_mass_response",
      better=dict(closest_to=1.0),
      x="truth.jet_pt",
      binning=JET_PT,
  )
  builds(Mean, name="n_particles_reco", variable="reco.n_particles")
  ```
  Metrics: `Median`, `IQR`, `Mean`, `Std`, `Quantile`, `Efficiency`, `FakeRate`
  (`scimo_bench/scorer.py`). A plain observable name without `truth.` / `reco.` is an error.

---

## Defining cuts

Cuts run after the join, before anything is computed, and the same cutflow applies to every
model. Every step states **which side it cuts** with `on`:

| `on` | Cuts | Example use |
|------|------|-------------|
| `truth` | the truth objects / events only | the truth acceptance in which efficiency is measured |
| `reco` | the models' output objects only | a threshold on what the models reconstruct |
| `both` | truth and reco, each on its own values | the same acceptance on both sides |

- **Object cuts** (`min_pt`, `max_abs_eta`) remove objects and may be `truth`, `reco` or
  `both`. With `both`, a reco particle is cut on its reconstructed pT, a truth particle on
  its true pT.
- **Event cuts** (`min_particles`) remove whole events and must be `on: truth`: every model is
  scored on the same events, so an event cannot be dropped for one model and kept for another
  (anything else is a configuration error).
- Cutting one side only changes what the metrics mean: with `min_pt` on `truth` only, a
  reco particle below the threshold has no truth partner left and counts as a fake.

The cutflow `preselection` (`scimo_bench/configs/cutflows.py`):

```python
cutflow_store(
    make_config(
        name="preselection",
        steps=[
            step("max_abs_eta", 2.5, on="both"),
            step("min_pt", 1.0, on="both"),
            step("min_particles", 1, on="truth"),
        ],
    ),
    name="preselection",
)
```

A step cuts the collection `particles` unless it says otherwise
(`step("min_pt", 2.0, on="truth", collection="tracks")`). Add a cutflow the same way (one
`cutflow_store(...)` with a new name, then `cutflow=<name>`), or change one on the command
line: `cutflow=preselection cutflow.steps.1.value=2.0 cutflow.steps.1.on=truth`.

A new kind of cut: a function `(objects, value) -> keep mask` in `scimo_bench/cutflow.py`;
`objects` is the step's collection, the mask is per object or (`level="event"`) per event:

```python
@cut(level="object")
def max_pt(objs, value):
    """Keep objects with pT below the value."""
    return objs.pt < value
```

`cutflow.csv` lists every step with its `level`, its side `on`, and the events, objects and
dropped objects for the truth and for each model.

---

## Plots

Every plot is one function in the dataset's `plots.py`, saved as
`plots/<function name>.png/.pdf`:

```python
@plot
def jet_mass(p):
    fig, ax, rax = p.figure(ratio=True)  # p.figure() without a ratio panel: fig, ax
    p.hist(ax, "jet_mass", np.linspace(0, 100, 51), ratio_ax=rax)
    p.legend(ax)
    return fig
```

`p` (`scimo_bench/plotter.py`, `PlotContext`) has the run's files and the helpers:

| Helper | Draws |
|--------|-------|
| `p.hist(ax, name, bins, ratio_ax=None, truth=True, models=None)` | an observable: truth (grey) and each model's reco output (lines), ratio reco / truth; a comparison: one line per model |
| `p.metric_vs_x(ax, metric, log_x=False)` | a binned metric from `report.csv` against its x, with interval bands |
| `p.legend(ax)`, `p.log_x(ax)`, `p.figure(...)` | legend above the axes, log axis, figure in the run's style |
| `p.values(side, name)`, `p.models`, `p.styles[model]`, `p.label(model)`, `p.report` | the raw material for anything custom (plain matplotlib on `ax`) |

**Truth or reco is always on the plot.** Legends say `truth` and `<model> (reco)`; the x axis
says `(truth)` or `(reco)` when only one level is drawn (e.g. `tracks per event (truth)`);
comparison labels name both (`E_reco / E_truth`); a metric's x axis names its side
(`jet pT (truth)`). A plot without data is skipped with a warning.

Which plots a run draws: the task's `plots` list, plus `plots.extra=[...]`, minus
`plots.skip=[...]` (or `plots.draw=[...]` to replace the list). Binning, scales and labels
live in the function. Generic plots: `ranking`, and quality vs cost with `plots=pareto`.
`plots.style` sets the look of all plots (palette, line styles, sizes, fonts, any matplotlib
setting under `rc`); per model `models.<name>.color=red`, `.linestyle=--` (matplotlib colour
names).

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
│   ├── datasets/             eval datasets (cocoa/: reader, observables, comparisons, plots)
│   ├── protocol.py           the model protocol (the only interface to model code)
│   ├── predict.py            step predict: run a model, write its prediction file
│   ├── perf.py               runtime, timing scenarios, quality gate
│   ├── schema.py, io.py      prediction file format (contract v1), reading, join on event_id
│   ├── catalogue.py          the @observable, @comparison, @plot decorators
│   ├── cutflow.py            cuts (plain functions on a collection)
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
| `dataset` | truth reader, package of observables/comparisons/plots, required fields | `cocoa` |
| `models` | from your models file | yours |
| `runtime` | device, precision, compile, batch size for model code | `gpu`, `cpu` |
| `perf` | scenarios, batch sizes, warmup, repeats | `single`, `offline`, `both`, `smoke` |
| `cutflow` | cuts applied to every model | `preselection` |
| `plots` (+ `plots/pareto`) | which plots, style, Pareto plot | `default`, `pareto` |
| `output` | run folder, bootstrap size, warn or stop on gaps | `default` |

A new cluster is one `site_conf(...)` entry in `configs/sites.py` (Slurm); a new task is
one entry in `configs/tasks.py`.

### Design rules

The quality path never runs models; model code runs only through the protocol, in jobs.
Every physics quantity is one function of its dataset (an observable or a comparison);
scorer and plotter compute none.
All models are scored on the same events, cuts and binning. A gap for one model never
aborts the run. Results are files, and plots can be redrawn from them. Details in
[`docs/DESIGN.md`](docs/DESIGN.md) and `CLAUDE.md`.

---

## Development

```bash
pixi run test            # all tests, in-process (~2 min); real-data tests skip without /ptmp
pixi run test-job raven  # the same tests as a CPU job
pixi run lint            # ruff check + format check
pixi run precommit       # all pre-commit hooks (ruff, yamlfix, yamllint); `pixi run pre-commit install` once per clone
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
