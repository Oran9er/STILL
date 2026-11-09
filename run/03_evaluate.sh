#!/bin/bash
# Evaluates a trained verifier on SD-V, SD-K and RC-3D: ./run/03_evaluate.sh <run-name> [model path]
source "$(dirname "$0")/common.sh"
n=${1:?run name}; M=${2:-runs/$1/final}

# SD-V: scores on dev / test / controls; the exponent a and the threshold are chosen on dev
for s in dev test ctl; do two_view $M ${n}sd vsd_$s $s; done
$PY_VLM scripts/fuse_sdiff.py ${n}sd | tee logs/fuse_${n}sd.json

# SD-K at the SD-V operating point
two_view $M ${n}sdk vsd_sdk_test test
$PY_VLM scripts/sdk_at_sdv_op.py $n

# RC-3D AP50 (present view, absent view, both)
two_view $M ${n}rc vsd_rc3d test
$PY_VLM scripts/rc3d_from_tv.py $n
$PY_VLM scripts/rc3d_project.py res/${n}_rc3d_Vg.jsonl res/${n}P_rc3d_Vg.jsonl
$PY_VLM scripts/rc3d_ap.py res/${n}P_rc3d_Vg.jsonl
