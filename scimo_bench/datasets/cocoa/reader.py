"""COCOA truth from the raw ROOT files.

Tree `EventTree`, one entry per event, event id `eventNumber`. Two collections (GeV, rad):
- `particles` (`particle_*` branches): the truth particles, what models reconstruct;
- `tracks` (`track_*` branches): reconstructed tracks, a model input (models do not
  output them, so observables of tracks exist for the truth only).
"""

from __future__ import annotations

import os

import awkward as ak
import numpy as np
import uproot

from scimo_bench import io
from scimo_bench.schema import EVENT_ID

EVENT_BRANCH = "eventNumber"
COLLECTIONS = {
    "particles": {
        "pt": "particle_pt",
        "eta": "particle_eta",
        "phi": "particle_phi",
        "e": "particle_e",
        "pdgid": "particle_pdgid",
    },
    "tracks": {
        "pt": "track_pt",
        "eta": "track_eta",
        "phi": "track_phi",
        "d0": "track_d0",
        "z0": "track_z0",
        "particle_idx": "track_particle_idx",
    },
}


class CocoaTruth:
    """Reads COCOA raw ROOT truth → `event_id` + `particles` + `tracks`."""

    def __init__(self, path: str | os.PathLike, tree: str = "EventTree"):
        self.path = path
        self.tree = tree

    def read(self, event_ids: np.ndarray | None = None) -> ak.Array:
        """The file's events (only `event_ids` if given), in file order."""
        branches = [b for fields in COLLECTIONS.values() for b in fields.values()]
        with uproot.open(io.resolve_path(self.path)) as f:
            raw = f[self.tree].arrays([EVENT_BRANCH, *branches])
        if event_ids is not None:
            raw = raw[np.isin(ak.to_numpy(raw[EVENT_BRANCH]), np.asarray(event_ids))]
        out = {EVENT_ID: ak.values_astype(raw[EVENT_BRANCH], np.int64)}
        for name, fields in COLLECTIONS.items():
            out[name] = ak.zip({k: raw[b] for k, b in fields.items()})
        return ak.zip(out, depth_limit=1)
