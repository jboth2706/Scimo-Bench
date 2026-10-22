"""COCOA single-jet dataset.

- `reader.py`: the truth from the raw ROOT file (`particles`, `tracks`);
- `observables.py`: quantities of one set of events (truth, or a model's reco output);
- `comparisons.py`: quantities of truth and one model (jet responses, ΔR, multiplicity);
- `plots.py`: one function per plot.
"""

from scimo_bench.catalogue import Catalogue
from scimo_bench.datasets.cocoa import comparisons, observables, plots

CATALOGUE = Catalogue.collect(observables, comparisons, plots)
