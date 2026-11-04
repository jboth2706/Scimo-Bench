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
| continuous data (`pflow_cont` model input) | `/ptmp/jboth/COCOA/Continuous_7mod/{train,val}` (same path on Viper); `topo_cont`, `track_cont`: 5 features per object (eta, phi in columns 3, 4), standardized with train's `*_cont_meta.npz`; truthpart as tokens; val 99,954 events (all 20,000 task events), `event_numbers.npy` raw int64 |
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

## Continuous-input nanoHEP (`pflow_cont`)

- Code in hep4m_scimo (branch `exp/cont_inputs`): `modalities/cont_token_stream.py` (layout,
  vocabulary with `<CONT:m>` / `<CONT_POS>` placeholders, encoder, dataset),
  `eval/cont_token_stream_decode.py` (`load_event_chunks`, `decode_chunk`),
  `configs/experiments/nanohep_cont.py` (`pflow_cont/{base,smoke,20M_10ep}`),
  `scripts/decode_cont.py` (reference decoding of a run).
- `data.split_pos=true`: every topo cluster / track takes two positions (content features,
  then (eta, sin phi, cos phi) at `<CONT_POS>`).
- Trained on Viper (MI300A, `workflow/viper/submit.sh`, outputs on Viper's `/ptmp`, which
  is `/viper/ptmp2/jboth` from the compute nodes, not visible from Raven): benchmark there
  with `site=viper models_file=nanohep_viper`, or copy the run folder to Raven.
- Viper setup for `site=viper`: model env `/viper/u2/jboth/scimo-rocm` (untouched; pyarrow
  for it in `/viper/u2/jboth/scimo-bench-env/rocm-extras`), default env
  `/viper/u2/jboth/scimo-bench-env` (copy of `pixi.toml`/`pixi.lock`, `pixi install -e
  default --frozen`), truth file copied to `/viper/ptmp2/jboth/COCOA/COCOA_samples_singlejet/`.
  On the login nodes `/u/jboth` and `/ptmp/jboth` are not the compute nodes' (`/viper/u2`,
  `/viper/ptmp2`); pixi is `/viper/u2/jboth/.pixi/bin/pixi`.
- SciMo-Bench class: `examples/nanohep_cont_model.py` (model type `nanohep_cont`).
