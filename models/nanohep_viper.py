"""Models file: nanoHEP runs on Viper (models_file=nanohep_viper, site=viper).

    ./run.sh site=viper run=20M_10ep models_file=nanohep_viper task=pflow \
    'models=[tok_n5_20M,cont_n5_20M,cont_split_n5_20M,cont_binned_n5_20M,cont_binnedsum_n5_20M]' \
        steps=[predict,perf,score,plot] perf=single \
        cutflow=preselection

Viper's compute nodes see the data under /viper/ptmp2 (not the login node's /ptmp/$USER),
so checkpoint, event store, VQ-VAE and modality dict are given as such paths.
"""

from scimo_bench.configs.models import add_model

PTMP = "/viper/ptmp2/jboth"
RUNS = f"{PTMP}/runs/HEP4M-20M-10ep"
STORE = f"{PTMP}/COCOA/Continuous_7mod"  # continuous features and topo/track/truthpart tokens
DECODER = dict(
    vqvae_root=f"{PTMP}/COCOA/Checkpoints",
    modality_dict="/viper/u2/jboth/nanoHEP/experiments/4m7_gold/modality_dict_pflow.yml",
)

add_model(
    "tok_n5_20M",
    type="nanohep",
    label="Method1",
    checkpoint=f"{RUNS}/nanohep_tok_n5_20M_10ep/checkpoints/final.ckpt",
    options=dict(tokenized_root=STORE, **DECODER),
)
add_model(
    "cont_n5_20M",
    type="nanohep_cont",
    label="Method2",
    checkpoint=f"{RUNS}/nanohep_cont_n5_20M_10ep/checkpoints/final.ckpt",
    options=dict(data_root=STORE, **DECODER),
)
add_model(
    "cont_split_n5_20M",
    type="nanohep_cont",
    label="Method3",
    checkpoint=f"{RUNS}/nanohep_cont_split_n5_20M_10ep/checkpoints/final.ckpt",
    options=dict(data_root=STORE, **DECODER),
)
add_model(
    "cont_binned_n5_20M",
    type="nanohep_cont",
    label="Method4",
    checkpoint=f"{RUNS}/nanohep_cont_binnedpos_n5_20M_10ep/checkpoints/final.ckpt",
    options=dict(data_root=STORE, **DECODER),
)
add_model(
    "cont_binnedsum_n5_20M",
    type="nanohep_cont",
    label="Method5",
    checkpoint=f"{RUNS}/nanohep_cont_binnedsum_n5_20M_10ep/checkpoints/final.ckpt",
    options=dict(data_root=STORE, **DECODER),
)
