# Shared helpers for the run/*.sh scripts.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
mkdir -p res logs data/p2
G=(${GPUS//,/ })
P=data/p2

# pi^3 places for one query file, sharded over the GPUs, merged into data/p2/pi3_anchor_<set>.json
pi3() {
  local q=$1 n=${#G[@]}
  for s in $(seq 0 $((n - 1))); do
    CUDA_VISIBLE_DEVICES=${G[$s]} $PY_SD -I scripts/pi3_anchor.py --queries $P/items_$q.jsonl --shard $s/$n > logs/pi3_${q}_s$s.log 2>&1 &
  done; wait
  $PY_SD -c "import json, glob; d = {}
for f in sorted(glob.glob('$P/pi3_anchor_${q}_s*.json')): d.update(json.load(open(f)))
json.dump(d, open('$P/pi3_anchor_$q.json', 'w')); print('$q places', len(d))"
  rm -f $P/pi3_anchor_${q}_s*.json
}

# Verifier scores for one item file: sv <model> <tag> <items-set> <out-name> [score_ver options]
sv() {
  local m=$1 t=$2 set=$3 out=$4 n=${#G[@]}; shift 4
  for s in $(seq 0 $((n - 1))); do
    CUDA_VISIBLE_DEVICES=${G[$s]} $PY_VLM scripts/score_ver.py --model $m --tag $t --items $P/items_$set.jsonl --shard $s/$n "$@" > logs/sv_${t}_${set}_s$s.log 2>&1 &
  done; wait
  cat res/${t}_${set}_s*.jsonl > res/${t}_$out.jsonl && rm -f res/${t}_${set}_s*.jsonl
}

# Scene view + place view for one set, combined into res/<tag>_tv_<out>.jsonl
two_view() {
  local m=$1 t=$2 set=$3 out=$4
  sv $m $t $set ver_$out
  sv $m $t cropP_$set crop_$out --bs 32 --full
  $PY_VLM scripts/combine_views.py res/${t}_ver_$out.jsonl res/${t}_crop_$out.jsonl res/${t}_tv_$out.jsonl
}

# SceneDiff change regions for one item file: sdiff <items-file>
sdiff() {
  local f=$1 name=$(basename $1 .jsonl); name=${name#items_}; local n=${#G[@]}
  for s in $(seq 0 $((n - 1))); do
    CUDA_VISIBLE_DEVICES=${G[$s]} $PY_SD -I scripts/run_scenediff.py --items $f --tag SDIFF --shard $s/$n > logs/sdiff_${name}_s$s.log 2>&1 &
  done; wait
  cat res/SDIFF_${name}_Vg_s*.jsonl > res/SDIFF_${name}_Vg.jsonl && rm -f res/SDIFF_${name}_Vg_s*.jsonl
}
