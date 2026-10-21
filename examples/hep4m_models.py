"""Example models file: the nanoHEP runs of the HEP4M sweeps on Raven.

Copy it somewhere of your own and pass it with `models_file=<path>`:

    ./run.sh site=raven run=n6_sweep models_file=examples/hep4m_models.py task=pflow \
        'models=[nanohep_n6_adamw,nanohep_n6_muon]' steps=[predict,perf,score,plot] perf=single
"""

from scimo_bench.configs.models import add_model

RUNS = "/ptmp/jboth/runs"


def final(run: str) -> str:
    return f"{RUNS}/{run}/checkpoints/final.ckpt"


for opt, opt_label in (("adamw", "AdamW"), ("muon", "Muon")):
    # multi-epoch sweep (5 epochs over 6.4M events, n_layer 6), lr 1e-3
    add_model(
        f"nanohep_n6_{opt}",
        type="nanohep",
        label=f"nanoHEP n6 {opt_label}",
        checkpoint=final(f"HEP4M-multi-epoch/pflow_n6_6p4M_5ep_lr1e-3_{opt}"),
    )
    # depth scaling, lr 1e-3, completeP on
    for depth in (5, 6, 7, 8):
        add_model(
            f"nanohep_depth_n{depth}_{opt}",
            type="nanohep",
            label=f"nanoHEP depth n{depth} {opt_label}",
            checkpoint=final(
                f"HEP4M-muon-vs-adamW-compute-opt/pflow_depth_lr1e-3_n_layer{depth}_{opt}"
            ),
        )
