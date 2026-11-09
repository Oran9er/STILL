# Copy to scripts/env.sh and edit. Every run/*.sh script sources it.
export STILL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Python of the verifier environment (torch, transformers >= 5, see requirements.txt)
export PY_VLM=python
# Python of the SceneDiff environment (SceneDiff, pi^3, SAM, DINOv3; see the SceneDiff repository)
export PY_SD=python

# Base VLMs (Hugging Face ids or local paths)
export Q9=Qwen/Qwen3.5-9B
export Q3=Qwen/Qwen3-VL-8B-Instruct

# Four GPUs for training and scoring
export GPUS=0,1,2,3
