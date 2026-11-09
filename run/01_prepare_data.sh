#!/bin/bash
# Builds every item file used for training and evaluation. Needs the datasets in ext/ (see README).
source "$(dirname "$0")/common.sh"

# SceneDiff SD-V: frames, scene items (5 + 5 frames), first-vs-second-half controls
$PY_VLM -I scripts/build_sd.py
STILL_SDS_R=12 $PY_VLM -I scripts/build_sds.py
$PY_VLM -I scripts/build_sds.py
$PY_VLM -I scripts/build_sds_hc.py

# Grounding DINO boxes on every frame, then the verifier training queries
[ -f data/props_sds_all.json ] || echo '{}' > data/props_sds_all.json
CUDA_VISIBLE_DEVICES=${G[0]} $PY_VLM -I scripts/proposals.py --missing data/items_sds_train12.jsonl data/items_sds_test.jsonl data/items_sds_hc_train.jsonl data/items_sds_hc_test.jsonl
$PY_VLM -I scripts/build_verify.py trainN

# Dev split (10 scenes held out from training) and the protocol files in data/p2
$PY_VLM -I scripts/build_protocol2.py

# Places and place-view crops for the training queries; two-frame versions
pi3 verN_train
STILL_CROP_PI3=1 STILL_CROP_SETS=train $PY_VLM -I scripts/build_crops.py
$PY_VLM -I scripts/build_twoview.py

# KC-3D synthetic two-view pairs
$PY_VLM -I scripts/build_kc3d.py
CUDA_VISIBLE_DEVICES=${G[0]} $PY_VLM -I scripts/proposals.py --missing data/kc3d/items_kc3d_train.jsonl
$PY_VLM -I scripts/build_kc3d_queries.py
pi3 kc3d_train
STILL_CROP_PI3=1 STILL_CROP_SETS=kc3d $PY_VLM -I scripts/build_crops.py

# SD-V evaluation: SceneDiff proposals on dev / test / hard controls, their places and crops
sdiff $P/items_sds_dev.jsonl
sdiff $P/items_sds_test.jsonl
sdiff data/items_sds_hc_test.jsonl
$PY_VLM -I scripts/build_sdiff_queries.py
for q in vsd_dev vsd_test vsd_ctl; do pi3 $q; done
STILL_CROP_PI3=1 STILL_CROP_SETS=vsd $PY_VLM -I scripts/build_crops.py

# SD-K (kitchen subset, test only)
$PY_VLM -I scripts/build_sdk.py
sdiff $P/items_sdk_dev.jsonl
sdiff $P/items_sdk_test.jsonl
$PY_VLM -I scripts/build_sdiff_queries.py sdk
for q in vsd_sdk_dev vsd_sdk_test; do pi3 $q; done
STILL_CROP_PI3=1 STILL_CROP_SETS=vsd_sdk $PY_VLM -I scripts/build_crops.py

# RC-3D (two-view, test only)
$PY_VLM -I scripts/build_rc3d.py
sdiff data/rc3d/items_rc3d.jsonl
$PY_VLM -I scripts/build_sdiff_queries.py rc3d
pi3 vsd_rc3d
STILL_CROP_PI3=1 STILL_CROP_SETS=rc3d $PY_VLM -I scripts/build_crops.py
echo "data ready"
