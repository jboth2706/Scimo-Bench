# Design

Built on the "SciMo-Bench Draft" (Claude Docs, 12.10.2026) with three extensions: any model
family (boundary in physical space), a performance path, Pareto fronts. Configs are hydra-zen.

## Architecture

```
model side (any family)                       scimo-bench
  your repo: one protocol class ──────────┐  (docs/PROTOCOL.md)
                                          │  steps predict + perf (model code, GPU job)
                                          ├─> predictions (contract v1) ─┐   perf.csv
  or: prediction files handed in ─────────┘                              v
                                             io.py: load + join on event_id <── truth (raw data)
                                                        v
                                             cutflow.py ─> variables.py ─> scorer.py ─> report.csv
                                                                                        rankings.csv
                                                                    perf.csv ─(gate)─> report.csv
                                                                               plotter.py <┘
```

Steps (`steps=[...]`, run in this order): `predict`, `perf` (run model code through the
protocol), `score` (files only), `plot` (from the CSVs). Default `[score, plot]`.

## Modules (`scimo_bench/`)

| Module | Role |
|--------|------|
| `main.py` | entry: loads `models_file`, composes the config, runs the steps or submits jobs |
| `steps.py` | the four steps; each reads and writes files in the run folder |
| `configs/` | hydra-zen groups: `task`, `dataset` (+ `plot_specs`), `models`, `cutflow`, `plots`, `output`, `runtime`, `perf`, `site` |
| `datasets/` | eval datasets: a truth reader and a variable registry per dataset (`cocoa.py`) |
| `schema.py` | contract v1: field names, validation, `meta` dataclass |
| `io.py` | read predictions (Parquet), read truth (COCOA ROOT), join, dedupe, counts |
| `variables.py` | `Registry`, `@REGISTRY.variable(name, label=, reference=)`, helpers, generic variables |
| `cutflow.py` | `@cut(name, level=, variable=)` registry; masks on variables |
| `scorer.py` | statistics, binning, paired bootstrap, status, ranking, normalized score |
| `plotter.py` | `@register_plot(name)` registry; distributions, binned metrics, ranking, Pareto |
| `protocol.py` | the model protocol (the only interface to model code) |
| `predict.py` | step `predict`: run a protocol model over the task's events, write its file |
| `perf.py` | runtime, scenarios, timing, quality gate |
| `jobs.py` | login-node guard, job scripts, submission from a code snapshot |

## Config groups (hydra-zen)

Set per call (everything else has defaults):

| What | How |
|------|-----|
| where it runs, and default paths | `site=local` (in-process) / `site=raven` (Slurm jobs) |
| data root, prediction files, run folders, snapshots | `paths.data`, `paths.preds`, `paths.outputs`, `paths.snapshots` (defaults from the site) |
| which models | `models_file=<your file>` + `'models=[a,b]'` |
| what to do | `steps=[predict,perf,score,plot]` |
| task, cuts, plots | `task=pflow`, `cutflow=central`, `plots=default|pareto` |
| how models run | `runtime=gpu|cpu` (device, precision, compile, batch size), `perf=single|offline|both|smoke` |

Models live in the user's models file, not in this repo:

```python
from scimo_bench.configs.models import add_model

add_model("n6_muon", type="nanohep", checkpoint="/ptmp/.../final.ckpt")  # protocol class
add_model("hgpflow", family="regression", predictions="/ptmp/.../hgpflow")  # files only
```

A type (`files`, `nanohep`, or one added with `add_type`) gives defaults for the fields
`label, family, predictions, generation, model, checkpoint, options, env`. `predictions`
defaults to `${paths.preds}/<task>/<name>`. Each model is stored under its own package
`models.<name>`, so `models=[a,b]` merges them side by side.

Datasets (`configs/datasets.py`, group `dataset`) name the truth reader (`builds` of a class
in `datasets/`, path relative to `paths.data`), the variables module, the collections
predictions must have, and the plot catalogue, which composes into `plots.specs` (group
`plot_specs`). Tasks (`configs/tasks.py`) name their default dataset, the event list (a
file in this repo), parameters passed to the variables, metrics as `builds(Median, ...)`,
the plots drawn by default, ranking and the quality gate. `dataset=<name>` overrides the
task's dataset; the variables of the selected dataset are activated before the run.
`better` ∈ `lower | higher | {closest_to: v}`; used for rankings and Pareto dominance.
Resolvers `linspace`, `logspace` are registered when `scimo_bench.configs` is imported.

## Outputs

```
outputs/<task>/<YYYY-MM-DD_HH-MM-SS>/
├── resolved_config.yaml
├── run.log                 # git commit, input paths and sizes, warnings
├── report.csv              # model, metric, bin_low, bin_high, value, lower, upper, status,
│                           #   kind (quality|perf), task_version, condition
├── rankings.csv
├── cutflow.csv
├── per_event.parquet       # variable values per event after cuts (re-scoring without reload)
├── pareto.csv              # model, condition, x, y, intervals, on_front
└── plots/
```

`status` ∈ `ok | missing | invalid`. `condition` = hash of the generation config (quality)
or scenario + batch size + device (perf).

## Sites and jobs

Config group `site` (`configs/sites.py`): `local` runs a call in-process; a cluster site
(`raven`, Slurm) submits it and returns. The submitting call fixes the run folder, copies the
code and the models file to a snapshot folder, and writes `job-<steps>.sh` and `jobs.csv`
into the run folder. Model steps (`predict`, `perf`) run as one job on the runtime's device
in the models' pixi env; `score` + `plot` run as a CPU job that starts only if the model
job succeeded (`--dependency=afterok`). Each job re-runs the same overrides in-process from
the snapshot. `plot` alone runs in-process everywhere. On a login node of any known site,
outside a job, other steps are refused (`jobs.guard`). Folders a site lists as `node_local`
(raven: `/tmp`) are refused for jobs.

## Statistics

- Paired bootstrap over events (same resampled events for all models), 95 % interval,
  `n` from the output config (default 1000), fixed seed.
- Bins with zero denominator → `invalid`.
- Optional normalized score s = (m − m_base) / (m_ref − m_base), baseline and reference
  named in the task, both must be among the evaluated models.

## Performance path

| Scenario | Question | Measured |
|----------|----------|----------|
| `offline` | events/s at the best batch size | events/s, GPU-s per 1,000 events, peak memory |
| `single` | time for one event | latency p50 / p90 / p99 |
| `rate` (later) | throughput under a latency target | max events/s with p99 < target |
| `scaling` (later) | batch size / GPU scaling | throughput curves |

Generative extras: time to first object, time per object or per step, NFE per event.

Rules: warmup batches and compile time reported separately; timing with
`torch.cuda.synchronize()` / CUDA events; model time (inputs already on device) and end-to-end
time (with loading + decoding) are separate metrics; repeats with intervals; hardware
fingerprint; FLOPs per event and parameter count reported alongside; a perf value is
`invalid` unless the same generation config passes the task's `quality_gate`.

Model protocol (the only place model code is imported; `docs/PROTOCOL.md`):

```python
class Model(Protocol):
    def load(self, cfg: dict) -> None: ...  # checkpoint, generation, options, runtime
    def inputs(self, event_ids, size): ...  # model's own inputs, on device; not timed
    def generate(self, batch): ...  # timed
    def to_floats(self, out): ...  # event_id + particles in GeV, rad
    def info(self) -> dict: ...  # family, n_params, generation, flops/event
```

Named in the model config as `model="package.module:Class"`. A model without one gets
`missing` perf rows. Perf rows go ungated to `perf.csv`; the gate is applied when they are
merged into `report.csv`, so perf may run before scoring.

## Pareto front

Point = model × generation config, under one fixed perf condition (same hardware/batch for
all points). y = scalar quality metric (`kind=quality`), x = cost metric (`kind=perf`) or a
static cost (FLOPs/event, training compute). Join over model + `condition`. Dominance uses
`better` for y and "lower is better" for cost. Error bars from bootstrap and repeats;
optional robust dominance (intervals separated). Models with a quality/cost knob are drawn
as connected sweep curves. Output: plot + `pareto.csv` with `on_front`.

```python
store(
    builds(
        ParetoPlot,
        y="emd_median",
        x="latency_p50",
        perf_condition=dict(scenario="single", batch_size=1, device="A100"),
        x_log=True,
        connect_sweeps=True,
        robust_dominance=False,
    ),
    group="plots/pareto",
    name="emd_vs_latency",
)
```

## Model families

| Family | Generation config (meta) | Into the contract | Natural cost unit |
|--------|--------------------------|-------------------|-------------------|
| token AR (hep4m causal, nanoHEP) | argmax / temperature, top-k, multiplicity EOS/oracle | position bins → η/φ, VQ-VAE decoder → pT/E | tokens, forward steps per event |
| masked (hep4m masked) | masking, iterative unmasking steps | as token AR | forward passes per event |
| discrete diffusion | steps, schedule, sampler, temperature | as token AR once tokens are final | NFE per event |
| continuous (flows, diffusion in data space) | ODE/SDE solver, steps, tolerances | direct | NFE per event |
| regression | none | direct | one forward pass |

Common cost unit across families: events/s and GPU-s per 1,000 events on fixed hardware.

## Borrowed from LLM benchmarking

From memory, not re-checked against current versions.

- lm-evaluation-harness: task registry with versions; metrics with aggregation and
  `higher_is_better`; per-sample logs → `task.version`, `better`, `per_event.parquet`.
- HELM: many metrics per scenario incl. efficiency; one adapter per model family; matrix
  reports → quality and cost in one report; Pareto plots.
- MLPerf Inference: load generator separate from system under test; scenarios Offline /
  SingleStream / Server; performance only valid at a quality target → `perf.py`, `quality_gate`.
- LLM serving benchmarks: TTFT, inter-token latency, percentiles, warmup excluded →
  generative perf metrics.
- Inspect / OpenAI Evals: solver vs scorer, replayable logs → files + resolved config.
- Contamination checks → compare `event_id`s against the training event range (later).
