"""Models file for the tests (`models_file=tests/models.py`): model slots the tests point
at synthetic prediction files or the toy protocol model, and the dataset `synthetic`
(COCOA observables, comparisons and plots; truth from a contract Parquet file)."""

from hydra_zen import builds

from scimo_bench.configs.datasets import add_dataset
from scimo_bench.configs.models import add_model
from scimo_bench.datasets import ContractTruth

add_dataset(
    "synthetic",
    truth=builds(ContractTruth, path="???", populate_full_signature=True),
    module="scimo_bench.datasets.cocoa",
    collections=dict(particles=["pt", "eta", "phi", "e"]),
)

add_model("dummy")
add_model("second")
add_model("toy", model="tests.toy_model:ToyModel", family="regression")
add_model("nano", type="nanohep", checkpoint="/runs/n6/checkpoints/final.ckpt")
