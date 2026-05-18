#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/WorldPolicy/HY-WorldPlay}"
cd "$REPO_ROOT"

if [ -f "$REPO_ROOT/activate_worldplay.sh" ]; then
    source "$REPO_ROOT/activate_worldplay.sh"
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/local_models/hunyuanvideo_1_5}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/local_models}"
DEVICE="${DEVICE:-cuda}"
NEG_PROMPT="${NEG_PROMPT:-}"

python3 datasets/hy_preprocess/generate_neg_prompt_pt.py \
        --model_path "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --device "$DEVICE" \
        --neg_prompt "$NEG_PROMPT"
