#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

export T2V_REWRITE_BASE_URL="${T2V_REWRITE_BASE_URL:-<your_vllm_server_base_url>}"
export T2V_REWRITE_MODEL_NAME="${T2V_REWRITE_MODEL_NAME:-<your_model_name>}"
export I2V_REWRITE_BASE_URL="${I2V_REWRITE_BASE_URL:-<your_vllm_server_base_url>}"
export I2V_REWRITE_MODEL_NAME="${I2V_REWRITE_MODEL_NAME:-<your_model_name>}"

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
REWRITE="${REWRITE:-false}"
ENABLE_SR="${ENABLE_SR:-false}"
OUTPUT_PATH="${OUTPUT_PATH:-${ROOT_DIR}/outputs/ar_distilled_local}"

MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/local_models/hunyuanvideo_1_5}"
AR_DISTILL_ACTION_MODEL_PATH="${AR_DISTILL_ACTION_MODEL_PATH:-${ROOT_DIR}/local_models/ar_distilled_action_model/diffusion_pytorch_model.safetensors}"

if [[ ! -f "${AR_DISTILL_ACTION_MODEL_PATH}" ]]; then
  echo "Missing distilled action checkpoint: ${AR_DISTILL_ACTION_MODEL_PATH}" >&2
  exit 1
fi

required_dirs=(
  "${MODEL_PATH}/transformer/480p_i2v"
  "${MODEL_PATH}/vae"
  "${MODEL_PATH}/scheduler"
  "${MODEL_PATH}/text_encoder"
  "${MODEL_PATH}/vision_encoder"
)

missing_base=0
for d in "${required_dirs[@]}"; do
  if [[ ! -d "${d}" ]]; then
    echo "Missing base model directory: ${d}" >&2
    missing_base=1
  fi
done

if [[ "${missing_base}" -ne 0 ]]; then
  echo >&2
  echo "MODEL_PATH is incomplete. Set MODEL_PATH to your downloaded HunyuanVideo-1.5 base directory before running." >&2
  exit 1
fi

mkdir -p "${OUTPUT_PATH}"

torchrun --nproc_per_node="${N_INFERENCE_GPU}" "${ROOT_DIR}/hyvideo/generate.py" \
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
  --action_ckpt "${AR_DISTILL_ACTION_MODEL_PATH}" \
  --few_step true \
  --num_inference_steps 4 \
  --model_type ar \
  --use_vae_parallel false \
  --use_sageattn false \
  --use_fp8_gemm false \
  --transformer_resident_ar_rollout true \
  --width "${WIDTH}" \
  --height "${HEIGHT}"
