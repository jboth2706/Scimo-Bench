"""Dataset configs (group `dataset`).

A dataset says how to read its truth (`truth`: a reader built with `builds`), which
package holds its observables, comparisons and plots (`module`), and which collections and
fields every model must output (`collections`, `event_fields`).

Add a dataset with `add_dataset(...)` (here, or in a models file).
"""

from hydra_zen import builds, make_config, store

from scimo_bench.datasets.cocoa.reader import CocoaTruth

dataset_store = store(group="dataset")


def add_dataset(
    name: str,
    truth,
    module: str,
    collections: dict,
    event_fields=(),
    description: str = "",
) -> None:
    """Register dataset `name`: `truth` reader (a `builds(...)`), `module` (a package with
    a `CATALOGUE`, e.g. `scimo_bench.datasets.cocoa`), and the `collections` {name:
    [fields]} models must output."""
    dataset_store(
        make_config(
            name=name,
            description=description,
            truth=truth,
            module=module,
            collections=collections,
            event_fields=list(event_fields),
            zen_dataclass={"cls_name": f"Dataset_{name}"},
        ),
        name=name,
    )


# relative to `paths.data`
COCOA_VAL = (
    "${paths.data}/COCOA/COCOA_samples_singlejet/val_100K_cells256/"
    "singlejet_10_100kseg_bw100.0.root"
)

add_dataset(
    "cocoa",
    truth=builds(CocoaTruth, path=COCOA_VAL, populate_full_signature=True),
    module="scimo_bench.datasets.cocoa",
    collections=dict(particles=["pt", "eta", "phi", "e"]),
    description="COCOA single jet (val, 100k events), particles and tracks from the raw ROOT file",
)
