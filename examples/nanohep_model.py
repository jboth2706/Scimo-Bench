"""nanoHEP (token-stream GPT from hep4m_scimo) as a SciMo-Bench protocol model.

Register it in a models file (the `nanohep` model type points here):

    add_model("depth_n6_muon", type="nanohep",
              checkpoint="/ptmp/jboth/runs/<project>/<run>/checkpoints/final.ckpt")

What happens per batch of event ids:
- `inputs`: the events' decoding prefixes (topo + track tokens, then the cue for truthpart)
  from the tokenized data, read with hep4m_scimo's `load_event_chunks`;
- `generate`: autoregressive decoding of truthpart with `decode_chunk` (structural
  decoding: only truthpart codes, stops at <EOS>); this is what is timed;
- `to_floats`: tokens → (content, position) codes → physics with the truthpart VQ-VAE
  (`hep4m.decoding.Detokenizer` from the nanoHEP repo): pt, e [GeV], eta, phi [rad].

`options` (all have defaults for COCOA on Raven):
- `tokenized_root`: the tokenized data (default: the checkpoint's `data.tokenized_root`);
- `split`: tokenized split holding the task's events (default `val`);
- `modality_dict`: VQ-VAE configs and checkpoints (nanoHEP's `modality_dict_pflow.yml`);
- `vqvae_root`: value of `${NANOHEP_EXPERIMENTS_ROOT}` in that file;
- `max_new_tokens` (default 200: 32 particles × 6 tokens + <EOS>), `decode_batch_size`;
- `truth_tokens`: true → skip the model and decode the events' TRUE tokens (a closure
  check of the detokenization, not a model).

Runs in the `perf` environment (torch, hep4m_scimo, scimo, nanoHEP's `hep4m`).
"""

from __future__ import annotations

import os
from pathlib import Path

import hep4m_scimo.configs  # noqa: F401  (registers the configs' resolvers)
import numpy as np
import torch
from hep4m.decoding import Detokenizer
from hep4m_scimo.assembly import assemble_model
from hep4m_scimo.eval.token_stream_decode import decode_chunk, load_event_chunks
from hep4m_scimo.modalities.token_stream import (
    make_hep_vocab,
    token_stream_dataset,
    token_stream_runtime,
)
from omegaconf import OmegaConf
from scimo.core import TokenStreamEncoder

OUTPUT = "truthpart"
NANOHEP_REPO = Path("/raven/u/jboth/nanoHEP")
DEFAULTS = dict(
    split="val",
    modality_dict=str(NANOHEP_REPO / "experiments/4m7_gold/modality_dict_pflow.yml"),
    vqvae_root="/ptmp/jboth/COCOA/Checkpoints",
    max_new_tokens=200,
    decode_batch_size=256,
    truth_tokens=False,
)
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}


class NanoHEP:
    """The model protocol (`scimo_bench/protocol.py`) for a nanoHEP checkpoint."""

    def load(self, cfg: dict) -> None:
        """Model from the checkpoint's `hep4m_config`, tokenized data, VQ-VAE decoder."""
        rt = cfg["runtime"]
        self.device = rt["device"]
        self.dtype = DTYPES.get(rt.get("precision"), None) if "cuda" in self.device else None
        self.generation = dict(cfg.get("generation") or {})
        self.opts = {**DEFAULTS, **dict(cfg.get("options") or {})}
        self.checkpoint = cfg["checkpoint"]
        torch.manual_seed(int(self.generation.get("seed", 0)))

        ckpt = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        hcfg = OmegaConf.create(ckpt["hep4m_config"])
        model = assemble_model(hcfg, token_stream_runtime(hcfg.data, hcfg.get("vocab_pad_to", 1)))
        state = {
            k.removeprefix("model.").replace("._orig_mod.", "."): v
            for k, v in ckpt["state_dict"].items()
        }
        model.load_state_dict(state)
        self.model = model.to(self.device).eval()
        self.n_params = sum(p.numel() for p in model.parameters())

        data = OmegaConf.to_container(hcfg.data, resolve=True)
        io = data["sampling_dict"]["fixed_input_output_modalities"]
        self.input_mods, self.output_mods = list(io["input"]), list(io["output"])
        if self.output_mods != [OUTPUT]:
            raise ValueError(f"expected output modality {OUTPUT}, got {self.output_mods}")
        self.vocab = make_hep_vocab(**data["vocab_args"])
        self.encoder = TokenStreamEncoder(self.vocab, data["block_size"])
        root = self.opts.get("tokenized_root") or data["tokenized_root"]
        split_dir = Path(root) / self.opts["split"]
        self.dataset = token_stream_dataset(
            root, self.opts["split"], data["all_modalities"], self.vocab
        )
        numbers = np.fromfile(split_dir / "event_numbers.npy", dtype="<i8")
        self.index = {int(n): i for i, n in enumerate(numbers)}

        os.environ.setdefault("NANOHEP_EXPERIMENTS_ROOT", self.opts["vqvae_root"])
        self.detok = Detokenizer(self.opts["modality_dict"], [OUTPUT], device=self.device)

    def inputs(self, event_ids, size):
        """Decoding prefixes (and true codes) of these events, in order."""
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
        """truthpart tokens per event (global ids), by autoregressive decoding."""
        ids, events = batch
        if self.opts["truth_tokens"]:
            return ids, [ev.truth[OUTPUT] for ev in events]
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

    def _codes(self, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Global token ids → (content codes [n, 3], position codes [n, 3]); an
        incomplete last particle (cut by max_new_tokens) is dropped."""
        width = self.vocab.num_quantizers[OUTPUT] + self.vocab.num_q_pos[OUTPUT]
        tokens = tokens[: len(tokens) - len(tokens) % width]
        return self.vocab.decode_element_with_pos(OUTPUT, tokens)

    def to_floats(self, out):
        """Codes → pt, e [GeV], eta, phi [rad] with the truthpart VQ-VAE."""
        ids, codes = out
        n = np.array([len(c) for c, _ in codes])
        width = max(int(n.max(initial=0)), 1)
        content = np.zeros((len(codes), width, self.vocab.num_quantizers[OUTPUT]), np.int64)
        pos = np.zeros((len(codes), width, self.vocab.num_q_pos[OUTPUT]), np.int64)
        for i, (c, p) in enumerate(codes):
            content[i, : len(c)], pos[i, : len(p)] = c, p
        mask = torch.from_numpy(np.arange(width)[None, :] < n[:, None])
        dec = self.detok.decode_tokens(
            OUTPUT, torch.from_numpy(content), torch.from_numpy(pos), mask
        )
        keep = mask.numpy()
        phi = torch.atan2(dec[f"{OUTPUT}_sinphi"], dec[f"{OUTPUT}_cosphi"])
        cols = dict(
            pt=dec[f"{OUTPUT}_pt"],
            eta=dec[f"{OUTPUT}_eta"],
            phi=phi,
            e=dec[f"{OUTPUT}_e"],
        )
        particles = {k: [v.numpy()[i][keep[i]] for i in range(len(ids))] for k, v in cols.items()}
        return {"event_id": ids, "particles": particles}

    def info(self) -> dict:
        """family, parameters, the generation settings used, checkpoint."""
        return dict(
            family="token_ar",
            n_params=self.n_params,
            generation=self.generation,
            checkpoint=str(self.checkpoint),
        )
