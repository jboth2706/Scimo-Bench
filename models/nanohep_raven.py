"""Models file: nanoHEP runs on Raven (models_file=nanohep_raven).

    ./run.sh site=raven run=cont_split_n5 models_file=nanohep_raven task=pflow \
        'models=[tok_n5_20M,cont_split_n5_20M]' steps=[predict,perf,score,plot] perf=single \
        cutflow=preselection
"""

from scimo_bench.configs.models import add_model

# trained on Viper (project HEP4M-20M-10ep), run folders copied to Raven
add_model(
    "tok_n5_20M",
    type="nanohep",
    label="Method1",
    checkpoint="/ptmp/jboth/runs/HEP4M-20M-10ep/nanohep_tok_n5_20M_10ep/checkpoints/final.ckpt",
    # tokens from the continuous store (same events and objects as Tokenized_7mod)
    options=dict(tokenized_root="/ptmp/jboth/COCOA/Continuous_7mod"),
)
add_model(
    "cont_split_n5_20M",
    type="nanohep_cont",
    label="Method3",
    checkpoint="/ptmp/jboth/runs/HEP4M-20M-10ep/nanohep_cont_split_n5_20M_10ep/"
    "checkpoints/final.ckpt",
)
