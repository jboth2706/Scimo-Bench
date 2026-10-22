"""Models file: a nanoHEP run on Raven (models_file=nanohep_raven).

    ./run.sh site=raven run=depth_n6 models_file=nanohep_raven task=pflow \
        'models=[depth_n6_muon]' steps=[predict,perf,score,plot] perf=single \
        cutflow=preselection
"""

from scimo_bench.configs.models import add_model

add_model(
    "depth_n6_muon",
    type="nanohep",
    label="nanoHEP n6 Muon",
    checkpoint="/ptmp/jboth/runs/HEP4M-muon-vs-adamW-compute-opt/"
    "pflow_depth_lr1e-3_n_layer6_muon/checkpoints/final.ckpt",
)
