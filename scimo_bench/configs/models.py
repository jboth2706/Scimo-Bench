"""Model configs (group `models`) and model types.

The repo holds no experiment runs. Register yours in your own models file and pass it with
`models_file=<path>`; select several with `models=[a,b]`:

    from scimo_bench.configs.models import add_model
    add_model("n6_muon", type="nanohep", checkpoint="/ptmp/.../final.ckpt")

A model with a `model` class (`docs/PROTOCOL.md`) is run by scimo-bench (steps `predict`,
`perf`) in the pixi environment `env`; without one, only its prediction file is read.
"""

from hydra_zen import make_config, store

model_store = store(group="models")

# passed to the model's `load`, echoed by its `info()`, hashed into `condition`
ARGMAX = dict(mode="argmax", multiplicity="model", seed=0)

FIELDS = dict(
    label=None,  # name in plots; default: the model's name
    family=None,  # token_ar | masked | discrete_diffusion | continuous | regression
    predictions=None,  # default: ${paths.preds}/<task>/<name>
    generation={},
    model=None,  # protocol class "package.module:Class"
    checkpoint=None,
    options={},  # anything else the class's `load` reads
    env="default",  # pixi environment the model steps run in
    color=None,  # plot colour, a matplotlib name (default: a palette slot from the model name)
    linestyle=None,  # plot line style (default: from the sorted model names)
)

# defaults per model type; `add_type` adds one (e.g. in a models file)
TYPES = {
    "files": {},  # predictions handed in as files
    "nanohep": dict(
        family="token_ar", model="hep4m_scimo.bench:NanoHEP", generation=ARGMAX, env="perf"
    ),
}


def add_type(name: str, **defaults) -> None:
    """Register a model type: default field values for `add_model(..., type=name)`."""
    unknown = set(defaults) - set(FIELDS)
    if unknown:
        raise TypeError(f"model type {name}: unknown fields {sorted(unknown)}")
    TYPES[name] = defaults


def add_model(name: str, type: str = "files", **fields) -> None:
    """Register model `name` (fields in FIELDS; defaults from its `type`)."""
    if type not in TYPES:
        raise ValueError(f"model {name}: unknown type {type!r}; known: {sorted(TYPES)}")
    unknown = set(fields) - set(FIELDS)
    if unknown:
        raise TypeError(f"model {name}: unknown fields {sorted(unknown)}; known: {list(FIELDS)}")
    conf = {**FIELDS, **TYPES[type], **fields}
    conf["label"] = conf["label"] or name
    conf["predictions"] = conf["predictions"] or f"${{paths.preds}}/${{task.name}}/{name}"
    conf["generation"] = dict(conf["generation"])
    model_store(
        make_config(name=name, type=type, **conf, zen_dataclass={"cls_name": f"Model_{name}"}),
        name=name,
        package=f"models.{name}",
    )
