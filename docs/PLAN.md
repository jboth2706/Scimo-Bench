# MVP plan

## Goal

One command compares at least two real models on the task `pflow` (COCOA singlejet, val
split) with physics metrics and bootstrap intervals; a second measures their inference time;
a third draws the Pareto front of physics against cost. First models: nanoHEP n6 AdamW and
n6 Muon from the multi-epoch sweep; afterwards the depth scaling n5–n8.

```bash
./run.sh task=pflow 'models=[nanohep_n6_adamw,nanohep_n6_muon]' cutflow=central
./run.sh task=pflow 'models=[nanohep_n6_adamw,nanohep_n6_muon]' stage=perf perf=single_a100
./run.sh task=pflow 'models=[nanohep_n6_adamw,nanohep_n6_muon]' stage=plot plots=pareto
```

## Scope

**In the MVP**
- one task `pflow`, one truth source (COCOA raw ROOT)
- prediction contract v1 (particle collection, `event_id`, `meta`), Parquet
- one real producer: nanoHEP via a thin layer in hep4m_scimo (`hep4m_scimo/bench`),
  multiplicity from the model (EOS)
- variables, cuts, median / IQR / mean / std / efficiency / fake rate, paired bootstrap, rankings
- distribution, binned-metric, ranking and Pareto plots; multi-model overlay
- perf scenarios `offline` and `single` with one adapter

**After the MVP**
- perf scenarios `rate` and `scaling`, multi-GPU
- robust dominance, sweep curves for generation knobs (diffusion steps, temperature)
- more producers: hep4m causal / masked, HGPflow, discrete diffusion, continuous models
- task `sim`, contamination check, caching, ROOT output, hosted leaderboard

## Milestones

Dependencies: M0 → M1 → {M2, M3}; M3 → M4 → M5; M2 → M6; {M4, M5, M6} → M7.
Sizes are relative effort (S < M < L), not dates.

| # | Delivery | Done when | Size | Brief |
|---|----------|-----------|------|-------|
| M0 | scaffold: pixi, `run.sh`, hydra-zen main + store groups, resolvers, output folder | `./run.sh task=pflow models=[dummy]` composes, writes the output folder, exits 0; a misspelled `builds` argument fails at compose time | S | `tasks/M0.md` |
| M1 | `schema.py` (v1), `io.py` (predictions Parquet, COCOA truth, join), synthetic data | join on all 99,955 val events without loss; missing events counted; tests green | M | `tasks/M1.md` |
| M2 | nanoHEP class in hep4m_scimo implementing the model protocol (`docs/PROTOCOL.md`): load, inputs, generate, tokens → floats; run by `steps=[predict]` in a GPU job for the two n6 models | two prediction files; decoded truth tokens match the ROOT truth up to binning | L | `tasks/M2.md` |
| M3 | `variables.py`, `cutflow.py`, `cutflow.csv` | known-answer tests pass; EMD of identical sets is 0; cutflow equal for all models | M | `tasks/M3.md` |
| M4 | `scorer.py`: statistics, bins, paired bootstrap, status, ranking, normalized score, `report.csv`, `rankings.csv`, `per_event.parquet` | values match numpy references; missing field → `missing`, no abort; toy coverage ≈ 95 % | M | `tasks/M4.md` |
| M5 | `plotter.py`: distributions, binned metric, ranking; fixed model styles; `stage=plot` | plots for the two n6 models; redraw without predictions | S | `tasks/M5.md` |
| M6 | `perf.py`: model protocol, `offline` + `single`, timing rules, `quality_gate`; uses the M2 class | GPU job measures both models; repeats scatter < 5 %; failed gate → `invalid` | M | `tasks/M6.md` |
| M7 | `ParetoPlot`, `pareto.csv` | EMD median vs latency p50 for AdamW vs Muon (n6), then n5–n8 | S | `tasks/M7.md` |

M2 lives mostly in the hep4m_scimo repo (`/raven/u/jboth/scimo`); everything else here.

## Tests (all milestones)

- **Known answers:** predictions = truth (response 1, EMD 0, efficiency 1), shifted (known
  bias), smeared (known IQR), particles dropped (known efficiency).
- **Robustness:** a model without `particles.e`, one with missing events, one with empty bins.
- **Reproducibility:** same seed + config → identical `report.csv` (quality); perf within
  its scatter.
- **Integration:** end-to-end on 1,000 real val events on CPU (quality path only). Perf only
  as a GPU job, never on the login node.

## Decisions

Status: **proposal** = suggested, not yet confirmed by the user; **decided** = confirmed.
Ask before building on a proposal that matters for the current milestone.

| Question | Proposal | Status |
|----------|----------|--------|
| Repo location | this repo, `/raven/u/jboth/scimo-bench`, own pixi env; hep4m_scimo depends only on the contract | decided |
| Config system | hydra-zen (pure Python), no YAML | decided |
| Package name | `scimo_bench` (avoid clash with the `scimo` framework) | proposal |
| Prediction format | Parquet (awkward) in the MVP; ROOT later via `io.py` | proposal |
| Environments | `default` = quality path (CPU, no torch); feature `perf` = + pytorch and hep4m_scimo (editable `../scimo`) for adapters | proposal |
| Val subset | fixed list of 20,000 `eventNumber`s, versioned with the task | proposal |
| pT decoding | exported truthpart VQ-VAE decoder (`hep4m_scimo/scripts/export_decoders.py`); must run before M2 | proposal |
| Matching | ΔR < 0.05, greedy by pT; threshold in the task config | proposal |
| Duplicate truth particles | drop in `io.py`, count in the log (413 pairs in the val set) | proposal |
| Perf reference hardware | 1× A100 40 GB on Raven, bf16, compiled; batch sizes 1, 64, 256 | proposal |
| Multiplicity for nanoHEP | from the model (EOS), oracle only as an explicit, labelled variant | proposal |
