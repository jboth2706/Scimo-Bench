"""Models file: mm-flow on the WIS cluster (models_file=mmflow_wis), one model per transfer.

The truth and the event list are those of mm-flow's selection (its cuts and single-jet cut),
the first 20.5k events of the COCOA test file. On a GPU node (site=local, in-process; the
env needs torch and mm_flow, e.g. `PYTHONPATH=<Scimo-Bench>:<mm-flow>/src python -m
scimo_bench` instead of ./run.sh):

    PRED=/storage/agrp/dmitrykl/bench/predictions/NoRelEmaJoint50_RGLR
    ... run=mmflow models_file=mmflow_wis task=pflow 'models=[mmflow_tt2p,mmflow_c2p]' \
        steps=[predict,score,plot] cutflow=preselection \
        paths.outputs=/storage/agrp/dmitrykl/bench/outputs \
        dataset.truth.path=$PRED/truth.root task.events=$PRED/events.txt
"""

from scimo_bench.configs.models import add_model

RUN = (
    "/srv01/agrp/dmitrykl/projects/mm-flow/logs/rcfm_z_flow_test_COCOA_3M_Large_CellData_"
    "InCtxt_Cut4MLike_NoRel_JointEventHeadEMA_ReducedGrad_AdaLNSingleLowLR_20261106-T165126"
)
CHECKPOINT = f"{RUN}/ckpts/epoch=019-val_total_loss=0.98823.ckpt"
TEST = (
    "/storage/agrp/nilotpal/COCOA_for_HEPCLIP/run/Feb10_2026_testval/"
    "test_1M_cells256_isInfFalse/singlejet_0_100kseg_bw100.0.root"
)

add_model(
    "mmflow_tt2p",
    type="mmflow",
    label="mm-flow track+topo -> particle",
    checkpoint=CHECKPOINT,
    options=dict(path=TEST, n_events=21000, sources=["track", "topo"]),
)
add_model(
    "mmflow_c2p",
    type="mmflow",
    label="mm-flow cell -> particle",
    checkpoint=CHECKPOINT,
    options=dict(path=TEST, n_events=21000, sources=["cell"]),
)
