# Model protocol

How a model developer makes a model runnable by SciMo-Bench. You write **one class** in your
own repo; SciMo-Bench creates it inside its job, asks it for the task's events, times it,
writes its predictions to a file, and scores them against the truth. Your repo does not
need to import anything from SciMo-Bench (the protocol is structural).

If you cannot or do not want to do this, you can still hand in a prediction file
(`docs/CONTRACT.md`); SciMo-Bench then only scores it.

## The class

```python
class MyModel:
    def load(self, cfg: dict) -> None:
        """Build the model once. cfg: the model's config entry (checkpoint, generation,
        options, ...) plus cfg["runtime"] = {"device", "precision", "compile"}."""

    def inputs(self, event_ids, size):
        """The model's inputs for these events (np.int64 array of event ids), read from
        your own data, on the device. Not counted as model time."""

    def generate(self, batch):
        """Run the model on one batch. This call is what is timed (CUDA-synchronized)."""

    def to_floats(self, out):
        """generate()'s output as one row per event (see 'Output' below)."""

    def info(self) -> dict:
        """family, n_params, generation; optional checkpoint, flops_per_event,
        nfe_per_event, tokens_per_event."""
```

SciMo-Bench calls them in this order: `load` once; then, per batch of event ids,
`inputs` → `generate` → `to_floats`. For timing (`perf`) it also calls `generate` alone on
prepared batches, after a warmup.

## Output of `to_floats`

One row per event, for the task's collection. For the task `pflow` (particle flow):

| field | type | meaning |
|-------|------|---------|
| `event_id` | int, one per event | the event id you were given in `inputs` (COCOA `eventNumber`) |
| `particles.pt` | list of floats per event | transverse momentum, **GeV** |
| `particles.eta` | list of floats per event | pseudorapidity |
| `particles.phi` | list of floats per event | azimuth, **rad**, in (−π, π] |
| `particles.e` | list of floats per event | energy, **GeV** |

The units are the point: the floats must be on the same scale as the truth, not the
model's internal representation (token ids, bins, normalized or log values). Converting
back (detokenizing, inverting normalizations) is the class's job.

Accepted forms: an awkward array, or a dict such as

```python
{"event_id": ids, "particles": {"pt": [[...], ...], "eta": ..., "phi": ..., "e": ...}}
```

An event with no particles is an empty list. An event you return no row for counts as
missing for that model.

## `info()` and `generation`

`generation` is the sampling setup (e.g. `{"mode": "argmax", "multiplicity": "model",
"seed": 0}`). SciMo-Bench passes the configured one to `load` and checks that `info()`
reports the same; a mismatch is an error. It is hashed into the report's `condition`, which
pairs perf rows with quality rows for the quality gate. `multiplicity: oracle` (the model
is told the true particle count) is labelled in plots.

Use `generation["seed"]` for any randomness, so a rerun gives the same predictions.

## Registering and running

In your models file (outside this repo):

```python
from scimo_bench.configs.models import add_model

add_model(
    "my_model",
    model="my_package.bench:MyModel",
    family="token_ar",
    checkpoint="/path/to/final.ckpt",
    generation=dict(mode="argmax", seed=0),
    env="perf",
    options=dict(data_dir="/path/to/inputs"),
)
```

`env` is the pixi environment of this repo that the model steps run in; it must contain
your package and its dependencies (`perf` contains torch and hep4m_scimo).

```bash
./run.sh site=raven run=my_first_run models_file=~/my_models.py task=pflow 'models=[my_model]' \
    steps=[predict,perf,score,plot] perf=single
```

## Checklist

- `to_floats` returns floats on the truth's scale; φ wrapped to (−π, π].
- One row per requested event id; ids are never batch positions.
- `info()["generation"]` equals the configured `generation`.
- `load` honours `runtime` (device, precision, compile).
- No work for the next batch hidden in `inputs` that belongs to the model (it is not timed).
