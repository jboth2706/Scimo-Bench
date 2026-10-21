"""Tiny numpy models implementing the model protocol (`scimo_bench/protocol.py`).

`ToyModel` "predicts" the truth (pT and E scaled by `options.scale`) after a fixed amount of
matrix work, so its predictions have known answers and its timing is non-trivial.
"""

import awkward as ak
import numpy as np

from scimo_bench.datasets import ContractTruth
from scimo_bench.datasets.cocoa.reader import CocoaTruth

WIDTH, DEPTH = 32, 8


class ToyModel:
    def load(self, cfg):
        opts = dict(cfg.get("options") or {})
        self.generation = dict(cfg.get("generation") or {})
        if opts.get("report_generation") is not None:
            self.generation = dict(opts["report_generation"])
        self.scale = float(opts.get("scale", 1.0))
        self.w = np.random.default_rng(0).standard_normal((WIDTH, WIDTH)) / np.sqrt(WIDTH)
        self.truth = None
        if opts.get("truth"):
            reader = CocoaTruth if opts.get("source") == "cocoa" else ContractTruth
            self.truth = reader(opts["truth"]).read()
            ids = ak.to_numpy(self.truth.event_id)
            self.row = dict(zip(ids.tolist(), range(len(ids)), strict=True))

    def inputs(self, event_ids, size):
        ids = np.asarray(event_ids, dtype=np.int64)
        return ids, np.tile((ids % 7).astype(np.float64)[:, None], (1, WIDTH))

    def generate(self, batch):
        ids, x = batch
        for _ in range(DEPTH):
            x = np.tanh(x @ self.w)
        return ids, x

    def to_floats(self, out):
        ids, x = out
        if self.truth is None:
            return {"event_id": ids, "particles": {"pt": [[float(v)] for v in np.abs(x[:, 0])]}}
        p = self.truth.particles[[self.row[i] for i in ids.tolist()]]
        return {
            "event_id": ids,
            "particles": {
                "pt": p.pt * self.scale,
                "eta": p.eta,
                "phi": p.phi,
                "e": p.e * self.scale,
            },
        }

    def info(self):
        n = WIDTH * WIDTH
        return dict(
            n_params=n,
            flops_per_event=2 * n * DEPTH,
            family="regression",
            generation=self.generation,
        )


class NoInfoModel(ToyModel):
    def info(self):
        return {}


class Incomplete:
    def load(self, cfg):
        pass
