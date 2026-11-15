"""mm-flow (joint flow-matching model, github.com/FM-for-HEP/mm-flow) as a SciMo-Bench
protocol model.

Register it in a models file (the `mmflow` model type points here):

    add_model("mmflow_tt2p", type="mmflow",
              checkpoint="<mm-flow>/logs/<run>/ckpts/<epoch>.ckpt",
              options=dict(path="<COCOA ROOT file>", sources=["track", "topo"], n_events=21000))

The model generates the truth particles of an event from its source modalities (track +
topo, or cell): first the event features (multiplicity) with its event head, then the
particles with the flow. The data settings (inputs, cuts, single-jet cut) and the scale
dictionary are the training ones, from the checkpoint's run folder (`<run>/config.yaml`).
Per batch of event ids:
- `inputs`: the events from the input file (mm-flow's `MMDataset`), on the device;
- `generate`: event head and flow sampling (`EvaluationModel.generate_modality_batch`);
  this is what is timed;
- `to_floats`: unscaled particles: pt, e [GeV], eta, phi [rad] and the class `cls`.

`generation`: `sampler` (euler | dpm), `steps` (flow), `event_steps` (event head), `seed`.
`options`:
- `path`: the COCOA ROOT file holding the task's events (the model's inputs);
- `sources`: the source modalities, e.g. `["track", "topo"]` or `["cell"]`;
- `n_events`: events read from the start of the file; they must include the task's events.
  Events dropped by mm-flow's selection (more elements than `max_elems`) get no prediction;
- `use_ema` (default true): the checkpoint's EMA weights.
`runtime.precision` sets the float32 matmul precision (`fp32`: as mm-flow's evaluation);
`runtime.compile: false` turns off the compiled transformer layers.

Needs torch and the `mm_flow` package (e.g. `PYTHONPATH=<mm-flow>/src`) in the process that
runs the model.
"""

from __future__ import annotations

from pathlib import Path

import awkward as ak
import numpy as np
import torch
import yaml
from mm_flow.evaluation.config import DatasetConfig, EvaluationConfig, ModelConfig
from mm_flow.evaluation.model import EvaluationModel
from mm_flow.models.config import TransferParameters
from mm_flow.utils.data import MMDataset
from torch.utils.data import default_collate

from scimo_bench.datasets.cocoa.reader import CHARGED, NEUTRAL_HADRON, PHOTON

# mm-flow classes (charged hadron, electron, muon, neutral hadron, photon) → ours
CLASS_MAP = np.array([CHARGED, CHARGED, CHARGED, NEUTRAL_HADRON, PHOTON], dtype=np.int32)
MATMUL_PRECISION = {"fp32": "highest", "tf32": "high", "bf16": "medium"}


class MMFlow:
    """The model protocol (`scimo_bench/protocol.py`) for an mm-flow joint checkpoint."""

    def load(self, cfg: dict) -> None:
        """Model and input events, with the training data settings of the checkpoint's run."""
        opts = {"use_ema": True, **dict(cfg.get("options") or {})}
        gen = self.generation = dict(cfg["generation"])
        runtime = cfg["runtime"]
        self.device = torch.device(runtime["device"])
        torch.set_float32_matmul_precision(MATMUL_PRECISION[runtime["precision"]])

        self.checkpoint = Path(cfg["checkpoint"])
        run_dir = self.checkpoint.parent.parent
        data = yaml.safe_load((run_dir / "config.yaml").read_text())["data"]
        data_cfg = DatasetConfig.from_dict(
            data | dict(path=opts["path"], n_events=opts["n_events"])
        )
        self.transfer = TransferParameters(list(opts["sources"]), ["particle"])
        config = EvaluationConfig(
            eval_path=None,
            # the training scale dictionary, copied into the run folder
            scale_dict_path=str(run_dir / Path(data["scale_dict_path"]).name),
            dataset_config=data_cfg,
            model=ModelConfig(
                self.checkpoint,
                sampler_type=gen["sampler"],
                n_steps=gen["steps"],
                use_ema=opts["use_ema"],
                event_steps=gen["event_steps"],
            ),
            transfer_params=[self.transfer],
        )
        self.model = EvaluationModel(config).to(self.device)
        if not runtime["compile"]:
            self.model.model.net.joint_encoder.compile_layers = False

        self.dataset = MMDataset(
            file_path=data_cfg.path,
            inputs=data_cfg.inputs,
            event_inputs=data_cfg.event_inputs,
            scale_dict_path=config.scale_dict_path,
            num_events=data_cfg.n_events,
            max_elems=data_cfg.max_elems,
            sin_cos_transform=data_cfg.sin_cos_transform,
            scale_num=data_cfg.scale_num,
            cuts=data_cfg.cuts,
            single_jet_cut=data_cfg.single_jet_cut,
            sampling_seed=None,
        )
        numbers = ak.to_numpy(self.dataset.data_dict["event_number"])
        self.row = {int(n): i for i, n in enumerate(numbers)}
        torch.manual_seed(int(gen["seed"]))

    def inputs(self, event_ids, size):
        """(events, constituents, masks) of these events, on the device; events not in the
        dataset are skipped."""
        ids = np.array([e for e in map(int, event_ids) if e in self.row], dtype=np.int64)
        batch = default_collate([self.dataset[self.row[e]] for e in ids])[:3]
        return ids, [{k: v.to(self.device) for k, v in d.items()} for d in batch]

    def generate(self, batch):
        """Target event features (event head), then the particles; still scaled."""
        ids, (events, constituents, masks) = batch
        out = self.model.generate_modality_batch(
            events, constituents, masks, self.transfer, to_cpu=False
        )
        return ids, out

    def to_floats(self, out):
        """Particles in physical units: pt, e [GeV], eta, phi [rad] and class `cls`."""
        ids, (events, constituents, masks) = out
        _, constituents = self.model.unscale_modalities(
            events, constituents, to_numpy=True, mask_dict=masks
        )
        p = constituents["particle"]
        mask = masks["particle"].cpu().numpy()
        cols = dict(pt=p["pt"], eta=p["eta"], phi=p["phi"], e=p["e"], cls=CLASS_MAP[p["class"]])
        particles = {k: [v[i][mask[i]] for i in range(len(ids))] for k, v in cols.items()}
        return {"event_id": ids, "particles": particles}

    def info(self) -> dict:
        """family, parameters, the generation settings used, checkpoint."""
        return dict(
            family="continuous",
            n_params=sum(p.numel() for p in self.model.model.net.parameters()),
            generation=self.generation,
            checkpoint=str(self.checkpoint),
        )
