#!/bin/bash
# Trains the verifier: ./run/02_train.sh <run-name> <seed> [base model, default $Q9]
source "$(dirname "$0")/common.sh"
n=${1:?run name}; seed=${2:-0}; base=${3:-$Q9}
CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY_VLM -m torch.distributed.run \
  --nproc_per_node ${#G[@]} scripts/still_sft.py --model $base --out runs/$n --arm V --epochs 1 --seed $seed \
  --train $P/items_verN_train.jsonl $P/items_cropP_train.jsonl $P/items_ver2v_train.jsonl \
          $P/items_kc3d_train.jsonl $P/items_cropP_kc3d_train.jsonl 2>&1 | tee logs/train_$n.log
