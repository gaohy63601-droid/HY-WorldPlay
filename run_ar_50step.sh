#!/usr/bin/env bash
# Inference with HY-World1.5-Autoregressive-480P-I2V (50-step AR baseline).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

PROMPT="${PROMPT:-A paved pathway leads towards a stone arch bridge spanning a calm body of water. Lush green trees and foliage line the path and the far bank of the water. A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. The water reflects the surrounding greenery and the sky. The scene is bathed in soft, natural light, creating a tranquil and serene atmosphere. The pathway is composed of large, rectangular stones, and the bridge is constructed of light gray stone. The overall composition emphasizes the peaceful and harmonious nature of the landscape.}"
IMAGE_PATH="${IMAGE_PATH:-${ROOT_DIR}/assets/img/test.png}"
POSE="${POSE:-w-31}"
SEED="${SEED:-1}"
ASPECT_RATIO="${ASPECT_RATIO:-16:9}"
RESOLUTION="${RESOLUTION:-480p}"
NUM_FRAMES="${NUM_FRAMES:-125}"
WIDTH="${WIDTH:-832}"
HEIGHT="${HEIGHT:-480}"
N_INFERENCE_GPU="${N_INFERENCE_GPU:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
REWRITE="${REWRITE:-false}"
ENABLE_SR="${ENABLE_SR:-false}"
OUTPUT_PATH="${OUTPUT_PATH:-${ROOT_DIR}/outputs/ar_50step}"

MODEL_PATH="${MODEL_PATH:-/home/Yirenteam/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038}"
AR_ACTION_MODEL_PATH="${AR_ACTION_MODEL_PATH:-/home/Yirenteam/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/f4c29235647707b571479a69b569e4166f9f5bf8/ar_model/diffusion_pytorch_model.safetensors}"

mkdir -p "${OUTPUT_PATH}"

export MASTER_PORT="${MASTER_PORT:-29521}"
torchrun --master_port="${MASTER_PORT}" --nproc_per_node="${N_INFERENCE_GPU}" "${ROOT_DIR}/hyvideo/generate.py" \
  --prompt "${PROMPT}" \
  --image_path "${IMAGE_PATH}" \
  --resolution "${RESOLUTION}" \
  --aspect_ratio "${ASPECT_RATIO}" \
  --video_length "${NUM_FRAMES}" \
  --seed "${SEED}" \
  --rewrite "${REWRITE}" \
  --sr "${ENABLE_SR}" --save_pre_sr_video \
  --pose "${POSE}" \
  --output_path "${OUTPUT_PATH}" \
  --model_path "${MODEL_PATH}" \
  --action_ckpt "${AR_ACTION_MODEL_PATH}" \
  --few_step false \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --model_type ar
