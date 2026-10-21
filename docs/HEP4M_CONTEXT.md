# Context: hep4m_scimo, nanoHEP and the COCOA data

Facts the milestones rely on, as of 2026-10-20. Paths are on Raven (MPCDF). Verify before
relying on a line number; the code moves.

## Repos

| Repo | Path | Notes |
|------|------|-------|
| hep4m_scimo | `/raven/u/jboth/scimo` | GitHub `sambklein/hep4m_scimo`; branch `dev/hybrid-muon-adamW`; pixi env `.pixi/envs/default` (torch 2.9, CUDA) |
| scimo (framework) | `/raven/u/jboth/scimo-framework` | GitHub `sambklein/scimo`; Python package name `scimo` |

## Data

| What | Path |
|------|------|
| raw COCOA val (truth source) | `/ptmp/jboth/COCOA/COCOA_samples_singlejet/val_100K_cells256/singlejet_10_100kseg_bw100.0.root`, tree `EventTree`, 99,955 events |
| truth branches | `particle_pt`, `particle_eta`, `particle_phi`, `particle_e` (GeV), `particle_pdgid`, `eventNumber` |
| tokenized data (model input) | `/ptmp/jboth/COCOA/Tokenized_7mod/{train,val}`; raw uint16, 6 values per particle `[cb0, cb1, cb2, pos_eta, pos_cosphi, pos_sinphi]`, offsets int64, `event_numbers.npy` |
| VQ-VAE checkpoints (Stage 1) | `/ptmp/jboth/COCOA/Checkpoints/vqxtrpartx128x8x1xxx3hty86rc6e3c/` (truthpart) and siblings |
| HGPflow inference (possible baseline) | `/ptmp/jboth/COCOA/HGPflow_inference` (not inspected yet) |

Truth particles: pT ≥ 1 GeV, |η| < ~3, mean 6.8 per event (max 27). ΣE per event flat in
~20–200 GeV (single jet, uniform η within |η| < 2.5).

## Known data pitfalls

- 413 particle pairs in 400 val events are exact duplicates (identical pT, η, φ; mostly Σ, Λ,
  K⁰S with different `particle_old_idx`).
- In ~4 % of events the tokenized truthpart has one soft particle (pT < 2 GeV) fewer than
  the ROOT file; cause unknown.
- Tokenized truthpart is sorted by pT descending; the ROOT file is not.
- Positions are binned: 1024 bins, η ∈ [−3, 3], cos φ and sin φ ∈ [−1, 1]; bin centre =
  min + (id + 0.5) · width (`hep4m_scimo/data/npy_data_source.py: _decode_pos_tokens`,
  `scripts/pflow_report.py: _decode_pos_tokens`).

## Models to benchmark first (nanoHEP, token-stream GPT)

- Multi-epoch sweep (5 epochs over 6.4M events, n_layer 6, AR 32, head_dim 16, completeP off):
  `/ptmp/jboth/runs/HEP4M-multi-epoch/pflow_n6_6p4M_5ep_{,1gpu_}lr{3e-4,6e-4,1e-3}_{adamw,muon}/`
  (queued as of 2026-10-20; check `.train_done`).
- Depth scaling (lr 1e-3, completeP on, n_layer 5–8):
  `/ptmp/jboth/runs/HEP4M-muon-vs-adamW-compute-opt/pflow_depth_lr1e-3_n_layer{5..8}_{adamw,muon}/`
  (n8 still queued as of 2026-10-20).
- Each run dir: `config_t.yaml`, `checkpoints/final.ckpt` (contains `hep4m_config`, the
  resolved config), `.train_done`; project CSV `<project>/<project>.csv`.

## Producer building blocks in hep4m_scimo

| Need | Where |
|------|-------|
| rebuild a model from a checkpoint | `ckpt["hep4m_config"]` → `hep4m_scimo.assembly.assemble_model(cfg, runtime)`; runtime for nanoHEP: `hep4m_scimo.modalities.token_stream.token_stream_runtime(cfg.data, pad_to=cfg.vocab_pad_to)`; `state_dict` keys `model.*` (compile prefix already stripped) |
| decode nanoHEP | `hep4m_scimo/eval/token_stream_decode.py`: `load_event_chunks`, `decode_chunk` (argmax or temperature; `structural_decode`; stops at `<EOS>`) → global token ids; `vocab.decode_element_with_pos` → per-particle codes |
| causal hep4m generation | `hep4m_scimo/eval/generate.py: generate_truthpart` (greedy, fixed `n_particles`, no EOS stop) |
| pT from content codes | `hep4m_scimo/scripts/export_decoders.py` (TorchScript decoder + meta JSON); `scripts/pflow_report.py: load_decoder, decode_pt_batch, _inverse_transform_pt` |
| existing physics metrics to port | `scripts/pflow_report.py`: `_compute_jets` (vector sum, massless), `_compute_emd_event` (unbalanced EMD, ΔR cost capped at R=1) |

The existing report uses **oracle multiplicity** (truncates predictions to the true particle
count). SciMo-Bench defaults to the model's own multiplicity.

## Cluster

- No training / GPU work on the login node; sbatch on partition `gpu1` (1× A100, 18 CPUs,
  125 GB) or `gpu` (4× A100). `gpudev` (15 min) only for short tests.
- hep4m_scimo submit helpers: `workflow/raven/submit.sh`, `workflow/raven/sweep.sh`
  (`--dry-run`, `--dev`, `--resume`).
