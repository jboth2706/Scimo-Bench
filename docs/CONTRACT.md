# Prediction contract v1

The only interface between model code and SciMo-Bench. Producers (any model family) write
files in this format; the benchmark reads nothing else. Units are physical; there are no
tokens in required fields.

## File

- One Parquet file (awkward record array) per (model, task, generation config); optional
  sharding as `part-*.parquet` in one directory.
- Plus `meta.json` next to it (or Parquet key-value metadata `scimo_bench.meta`).

## Fields

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `event_id` | int64 [E] | yes | key into the truth; for COCOA the `eventNumber` branch / `event_numbers.npy`. Never a batch position. |
| `sample_id` | int32 [E] | no (default 0) | several draws per event for stochastic models; (`event_id`, `sample_id`) unique |
| `<collection>.<field>` | jagged float32 [E, n] | per task | objects in physical units, e.g. `particles.pt` [GeV], `particles.eta`, `particles.phi` [rad, (-π, π]], `particles.e` [GeV], optional `particles.cls` (int) |
| `<event_field>` | float32 [E] | per task | event-level quantities for regression models (e.g. `met` [GeV]) |
| `raw.*` | any | no | family-specific raw outputs for diagnostics (tokens, logits); the benchmark ignores them unless a diagnostic variable asks |

Which collections and fields a task needs is declared by the task. A missing optional field
is not an error: variables that need it report `missing`.

## `meta`

```json
{
  "schema_version": 1,
  "model": {"name": "nanohep_n6_muon", "family": "token_ar", "checkpoint": "<path>",
            "checkpoint_sha256": "<hex>", "n_params": 3423360},
  "generation": {"mode": "argmax", "temperature": null, "multiplicity": "model",
                 "steps": null, "seed": 0},
  "data": {"dataset": "cocoa_singlejet", "split": "val", "source": "<path>",
           "event_list": "<path or hash>"},
  "code": {"producer_repo": "hep4m_scimo", "producer_commit": "<sha>",
           "framework_commit": "<sha>"},
  "hardware": {"device": "A100-SXM4-40GB", "precision": "bf16-mixed", "compiled": true},
  "created": "2026-10-20T12:00:00Z"
}
```

- `family` ∈ `token_ar | masked | discrete_diffusion | continuous | regression`.
- `generation.multiplicity` ∈ `model | oracle`; `oracle` results must never be mixed with
  `model` results without being labelled.
- The generation block is hashed into the report's `condition` column.

## Rules

- `event_id` values must exist in the truth; unknown ids are counted and dropped with a warning.
- Objects with non-finite values are dropped per object and counted.
- Changing this file means bumping `schema_version`; `io.py` keeps reading older versions.
