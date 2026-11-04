"""nanoHEP with continuous inputs (hep4m_scimo `pflow_cont/*`) as a SciMo-Bench protocol model.

Register it in a models file (the `nanohep_cont` model type points here):

    add_model("cont_split_n5_20M", type="nanohep_cont",
              checkpoint="/ptmp/jboth/runs/<project>/<run>/checkpoints/final.ckpt")

The same token-stream GPT as `NanoHEP`, but topo clusters and tracks enter as continuous
features (`topo_cont`, `track_cont`; with `data.split_pos` their (eta, phi) in a second
position each; with `data.binned_pos` their (eta, cos phi, sin phi) bins as three position
tokens shared with truthpart, read from the store's `topo` / `track` token files, and with
`data.sum_pos_tokens` those three rows summed into one position) instead of tokens. The
truth particles are still generated as truthpart tokens, so `to_floats` (VQ-VAE codes →
physics) is `NanoHEP`'s. Per batch of event ids:
- `inputs`: the decoding prefixes (ids and features) from the continuous event store, read
  with hep4m_scimo's `cont_token_stream_decode.load_event_chunks`;
- `generate`: autoregressive decoding of truthpart (`cont_token_stream_decode.decode_chunk`,
  structural decoding, stops at <EOS>); this is what is timed;
- `to_floats`: as `NanoHEP`.

`options` (defaults for COCOA on Raven):
- `data_root`: the continuous event store (default: the checkpoint's `data.data_root`);
- `split`, `modality_dict`, `vqvae_root`, `max_new_tokens`, `decode_batch_size`,
  `truth_tokens`: as for `NanoHEP`.

Runs in the `perf` environment (torch, hep4m_scimo, scimo, nanoHEP's `hep4m`).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from hep4m_scimo.assembly import assemble_model
from hep4m_scimo.eval.cont_token_stream_decode import decode_chunk, load_event_chunks
from hep4m_scimo.modalities.cont_token_stream import (
    ContTokenStreamEncoder,
    cont_stream_layout,
    cont_token_stream_dataset,
    cont_token_stream_runtime,
)
from omegaconf import OmegaConf

from examples.nanohep_model import OUTPUT, NanoHEP

log = logging.getLogger(__name__)


class NanoHEPCont(NanoHEP):
    """The model protocol (`scimo_bench/protocol.py`) for a continuous-input nanoHEP."""

    def load(self, cfg: dict) -> None:
        """Model from the checkpoint's config, the continuous event store, VQ-VAE decoder."""
        self._setup(cfg)
        ckpt = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        hcfg = OmegaConf.create(ckpt.get("hep4m_config") or self._run_config())
        data = OmegaConf.to_container(hcfg.data, resolve=True)
        if self.opts.get("data_root"):
            data["data_root"] = self.opts["data_root"]
        runtime = cont_token_stream_runtime(data)
        self._set_model(assemble_model(hcfg, runtime), ckpt)
        for m, s in runtime["feature_stats"].items():  # the data's standardization == training's
            for k in ("mean", "std"):
                trained = getattr(self.model.content_emb, f"{k}_{m}").cpu().numpy()
                if not np.allclose(trained, s[k]):
                    raise ValueError(f"{m} {k}: checkpoint {trained} != data {s[k]}")

        io = data["sampling_dict"]["fixed_input_output_modalities"]
        self.input_mods, self.output_mods = list(io["input"]), sorted(io["output"])
        if self.output_mods != [OUTPUT]:
            raise ValueError(f"expected output modality {OUTPUT}, got {self.output_mods}")
        cont, tok = list(data["cont_modalities"]), list(data["token_modalities"])
        layout = cont_stream_layout(
            data["data_root"],
            cont,
            tok,
            data.get("feature_stats"),
            bool(data.get("split_pos")),
            bool(data.get("binned_pos")),
            bool(data.get("sum_pos_tokens")),
        )
        self.vocab, self.tok_vocab = layout.vocab, layout.vocab.base
        self.encoder = ContTokenStreamEncoder(
            self.vocab, layout.n_columns, data["block_size"], layout.pos_columns, layout.pos_tokens
        )
        self.dataset = cont_token_stream_dataset(
            data["data_root"],
            self.opts["split"],
            cont,
            tok,
            self.vocab,
            layout.feature_stats,
            pos_sources=list(layout.pos_tokens.sources.values()) if layout.pos_tokens else None,
        )
        self._set_index(Path(data["data_root"]) / self.opts["split"])
        self._set_detokenizer()

    def _run_config(self) -> dict:
        """The run folder's `config_t.yaml` (checkpoints without `hep4m_config`)."""
        path = Path(self.checkpoint).parent.parent / "config_t.yaml"
        with open(path) as fp:
            return yaml.safe_load(fp)

    def inputs(self, event_ids, size):
        """Decoding prefixes (ids, features) and true codes of these events, in order."""
        idxs = [self.index[int(e)] for e in event_ids]
        loader = load_event_chunks(
            self.dataset,
            self.encoder,
            idxs,
            self.input_mods,
            self.output_mods,
            chunk_size=len(idxs),
        )
        return np.asarray(event_ids, dtype=np.int64), next(iter(loader))

    def generate(self, batch):
        """truthpart codes per event, by autoregressive decoding."""
        ids, events = batch
        if self.opts["truth_tokens"]:
            nq, nqp = self.tok_vocab.num_quantizers[OUTPUT], self.tok_vocab.num_q_pos[OUTPUT]
            return ids, [
                (ev.truth[OUTPUT][0][:, :nq], ev.truth[OUTPUT][1][:, :nqp]) for ev in events
            ]
        n_long = sum(ev.prefix is None for ev in events)
        if n_long:
            log.warning("%d events: prefix longer than block_size, decoded as empty", n_long)
        mode = self.generation.get("mode", "argmax")
        with torch.autocast("cuda", self.dtype, enabled=self.dtype is not None):
            out = decode_chunk(
                self.model,
                self.vocab,
                events,
                self.output_mods,
                argmax=mode == "argmax",
                temperature=float(self.generation.get("temperature", 1.0)),
                structural_decode=True,
                max_new_tokens=int(self.opts["max_new_tokens"]),
                batch_size=int(self.opts["decode_batch_size"]),
                device=self.device,
            )
        return ids, [self._codes(o[OUTPUT].cpu().numpy()) for o in out]
