# CLAUDE.md — scimo-bench (SciMo-Bench)

Context for AI agents working in this repo. Humans: see README.md.

## What this repo is

SciMo-Bench evaluates the predictions of scientific foundation models against the truth,
and measures their inference cost. Configs (hydra-zen) define which models, data, cuts,
metrics and plots apply; each dataset defines its observables, comparisons and plots as plain
functions, and the core modules cut, compute metrics and draw. Several models run in one call and appear in the same plots. A fifth, optional
module measures inference performance. Quality and cost are combined on Pareto fronts.

M0, M1 and M3–M7 are built (status in README.md); M2 (the nanoHEP protocol class in
hep4m_scimo) and the GPU measurement are open.
**Continue milestone by milestone
following `docs/PLAN.md`; each milestone has its own brief in `docs/tasks/M<n>.md`.**

## Read first

| File | What it holds |
|------|---------------|
| `docs/PLAN.md` | MVP goal, scope, milestones M0–M7 with acceptance criteria, open decisions |
| `docs/DESIGN.md` | architecture, config groups, outputs, performance rules, Pareto front |
| `docs/CONTRACT.md` | the prediction contract v1 — the file format of predictions |
| `docs/PROTOCOL.md` | the model protocol — the class through which scimo-bench runs a model |
| `README.md` | per-call settings, datasets, variables, cuts, plots (how to add each) |
| `docs/HEP4M_CONTEXT.md` | facts about the first producer (hep4m_scimo / nanoHEP) and the COCOA data |
| `docs/tasks/M<n>.md` | one self-contained work order per milestone |

## Working agreement

- Work on **one milestone at a time**, in dependency order (`docs/PLAN.md`). Read its task
  file, implement, make its acceptance criteria pass, then stop and report.
- Every milestone ends with tests (`pixi run test`) and lint (`pixi run lint`) passing.
- If a task file and the design disagree, or a decision is marked open, **ask; do not guess.**
- Do not change `docs/CONTRACT.md` without being asked: other repos write files against it.
  Changing it means bumping `schema_version`.
- Commits: show the message to the user first, keep it short, no `Co-Authored-By` line.
  Do not push unless asked.

## Invariants (do not break)

1. **Configs are pure Python (hydra-zen).** `builds`, `make_config`, `store`; no YAML config
   files. The resolved config is still written as `resolved_config.yaml` per run.
2. **The model/benchmark boundary is in physical space.** The benchmark reads events and
   objects in physical units (GeV, rad) keyed by `event_id`. It never sees tokens,
   vocabularies or checkpoints. Model-family details live in `meta`.
3. **Quality path never runs models.** `io → cutflow → observables/comparisons → scorer → plotter` works on
   files only. Model code runs only through the model protocol (`protocol.py`, a class named
   in the model config), in the steps `predict` and `perf`, in jobs on a cluster.
4. **One source for physics.** Every quantity is a function of its dataset package
   (`datasets/<name>/`): an `@observable` (one set of events, run on truth and on every
   model) in `observables.py`, or a `@comparison` (truth and one model) in
   `comparisons.py`; shared helpers in `datasets/common.py`. Each plot is one `@plot`
   function in `plots.py`. `scorer.py` and `plotter.py` compute no physics themselves.
   Reading a dataset's truth format lives in its reader (`reader.py`), not in `io.py`.
   Truth vs reco (the models' output) is always explicit: cut `on`, metric `truth.`/`reco.`
   names, plot labels.
5. **All models on the same basis.** Same events (after the join), same cutflow, same binning,
   same normalization. Paired bootstrap over events.
6. **Gaps do not abort.** A missing field, metric or plot for one model → status `missing`
   (or `invalid` for empty denominators / failed quality gate), a warning, and the run goes
   on. Only an invalid configuration stops a run.
7. **Results are files.** `report.csv`, `rankings.csv`, `cutflow.csv`, `per_event.parquet`,
   `pareto.csv`, plots, `resolved_config.yaml`, `run.log` in one output folder per call.
   Plots can be redrawn from the CSVs (`steps=[plot]`).
8. **Performance numbers are honest.** Warmup and compile time reported separately;
   CUDA-synchronized timing; model time and end-to-end time are separate metrics; hardware
   fingerprint in `meta`; a perf value is `invalid` unless the same generation config passes
   the task's `quality_gate`.
9. **Package name is `scimo_bench`**, not `scimo` (that name is taken by the scimo training
   framework, which shares environments with the producers).

## Code style

- Imports at module top, never inside functions (ruff does not catch this; check by hand).
- Match the surrounding code; short docstrings; no speculative abstractions beyond the plan.
- Tests: pytest, synthetic data with known answers (`tests/synthetic.py` from M1 on).

## Environment and commands

```bash
pixi install                      # default env: quality path (CPU, no torch)
pixi run test                     # all tests (in-process; `pixi run test-job raven` as a job)
pixi run lint                     # ruff check + format check
pixi run precommit                # all pre-commit hooks (ruff, yamlfix, yamllint), as on commit
# models files: models/nanohep_raven.py (models_file=nanohep_raven), or a path
./run.sh site=raven run=<name> models_file=<file> task=pflow 'models=[a,b]' cutflow=preselection  # score+plot
./run.sh site=raven run=<name> models_file=<file> task=pflow 'models=[a,b]' \
    steps=[predict,perf,score,plot] perf=single                # GPU job, then CPU job
./run.sh site=raven run=<name> models_file=<file> task=pflow 'models=[a,b]' steps=[plot] plots=pareto
```

## Cluster rules (Raven, MPCDF)

- Never run training or GPU work on the login node. GPU measurements (M2 predictions, M6
  perf) run as sbatch jobs (1× A100, short time limits, code from a frozen snapshot).
  `site=raven` does this for every benchmark call (`scimo_bench/jobs.py`); without it the
  benchmark refuses every step but `plot` on a login node.
- Data lives on `/ptmp/jboth/...` (see `docs/HEP4M_CONTEXT.md`); do not copy it into the repo.
