#!/usr/bin/env bash
set -euo pipefail

# GameFactory/Minecraft dataset preprocessing

REPO_ROOT="${REPO_ROOT:-/workspace/WorldPolicy/HY-WorldPlay}"
cd "$REPO_ROOT"

if [ -f "$REPO_ROOT/activate_worldplay.sh" ]; then
    source "$REPO_ROOT/activate_worldplay.sh"
elif [ -d "$REPO_ROOT/.venv" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

echo "===================================="
echo "GameFactory/Minecraft Data Preprocessing"
echo "===================================="

# Set PYTHONPATH
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Input path. The directory should contain annotation.csv, metadata/, and video/.
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/datasets/gamefactory_raw}"

# Output paths
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/datasets/preprocessed_gamefactory_f129}"
OUTPUT_JSON="${OUTPUT_JSON:-dataset_index.json}"

# Model path
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/local_models/hunyuanvideo_1_5}"

# Target resolution
TARGET_HEIGHT="${TARGET_HEIGHT:-480}"
TARGET_WIDTH="${TARGET_WIDTH:-832}"

# Other options
DEVICE="${DEVICE:-cuda}"
NUM_SAMPLES="${NUM_SAMPLES:-}"  # Leave empty to process all segments; set a number for testing (e.g. NUM_SAMPLES=2)
TARGET_NUM_FRAMES="${TARGET_NUM_FRAMES:-129}"  # Empty=no resampling; e.g. 129 means index interpolation to 129 frames

echo ""
echo "Configuration:"
echo "  Dataset root: $DATA_ROOT"
echo "  Output dir: $OUTPUT_DIR"
echo "  Model path: $MODEL_PATH"
echo "  Target resolution: ${TARGET_WIDTH}x${TARGET_HEIGHT}"
echo "  Device: $DEVICE"
echo "  Num segments: ${NUM_SAMPLES:-all}"
echo "  Target frames: ${TARGET_NUM_FRAMES:-unchanged}"
echo ""

# Run preprocessing
python3 datasets/hy_preprocess/preprocess_gamefactory_dataset.py \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --output_json "$OUTPUT_JSON" \
    --model_path "$MODEL_PATH" \
    --target_height "$TARGET_HEIGHT" \
    --target_width "$TARGET_WIDTH" \
    --device "$DEVICE" \
    ${NUM_SAMPLES:+--num_samples $NUM_SAMPLES} \
    ${TARGET_NUM_FRAMES:+--target_num_frames $TARGET_NUM_FRAMES}
