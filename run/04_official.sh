#!/bin/bash
# SceneDiff official protocol: ./run/04_official.sh <run-name> [model path]
# Run 03_evaluate.sh first (it picks the operating point on SD-V dev).
source "$(dirname "$0")/common.sh"
n=${1:?run name}; M=${2:-runs/$1/final}; t=${n}off

# Official frames and SceneDiff's own predictions (once)
if [ ! -d ext/sd_pred/sdiff ]; then
  $PY_SD -I scripts/official_frames_ff.py
  for s in $(seq 0 $((${#G[@]} - 1))); do
    CUDA_VISIBLE_DEVICES=${G[$s]} $PY_SD -I scripts/run_official.py --out ext/sd_pred/sdiff --shard $s/${#G[@]} > logs/official_sdiff_s$s.log 2>&1 &
  done; wait
fi

# Verify every SceneDiff object and write its masks with the fused scores
[ -f $P/items_off_test.jsonl ] || { $PY_SD -I scripts/official_rescore.py build; pi3 off_test
  STILL_CROP_PI3=1 STILL_CROP_SETS=off $PY_VLM -I scripts/build_crops.py; }
two_view $M $t off_test test
read a thr < <(python3 -c "import json; d = json.load(open('logs/fuse_${n}sd.json')); print(d['a'], d['thr'])")
$PY_SD -I scripts/official_rescore.py write res/${t}_tv_test.jsonl $a ext/sd_pred/ours_$n
$PY_SD -I scripts/official_rescore.py write res/${t}_tv_test.jsonl $a ext/sd_pred/ours_${n}_thr --thr $thr

# Official metrics (SceneDiff's evaluate_multiview.py); px/im IoU is read from the thresholded run
cd ext/scene_diff_code
for pred in ours_$n ours_${n}_thr; do for set in varied kitchen; do
  $PY_SD -I scripts/evaluate_multiview.py --pred_dir ../sd_pred/$pred --splits test --sets $set \
    --output_path ../sd_pred/eval_${pred}_$set.json --visualize False
done; done
