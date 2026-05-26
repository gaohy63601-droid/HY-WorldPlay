#!/usr/bin/env bash
# DMD self-forcing init-phase training launcher.
# 还原 pod 上跑出 outputs/dmd_init_pod_fixed/checkpoint-{100,200,300} 的命令。
# 所有变量都可被外部 env 覆盖;local01 上用:
#   REPO_ROOT=/raid/yiren/ghy/new/HY-WorldPlay \
#   HF_HUNYUAN=/home/Yirenteam/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038 \
#   HF_HY=/home/Yirenteam/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/f4c29235647707b571479a69b569e4166f9f5bf8 \
#   TRAIN_JSON_PATH=/raid/yiren/ghy/new/HY-WorldPlay/datasets/preprocessed_gamefactory_f129/dataset_index.json \
#   OUTPUT_DIR=/raid/yiren/ghy/new/HY-WorldPlay/outputs/dmd_init_local \
#   CUDA_VISIBLE_DEVICES=4,5 NUM_GPUS=2 \
#   bash run_dmd_train.sh
set -eo pipefail

# ---- 路径 (pod 默认值) ----
export REPO_ROOT="${REPO_ROOT:-/workspace/ghy/HY-WorldPlay}"
export HF_HUNYUAN="${HF_HUNYUAN:-/workspace/ghy/models/HunyuanVideo-1.5}"
export HF_HY="${HF_HY:-/workspace/ghy/models/HY-WorldPlay}"
export TRAIN_JSON_PATH="${TRAIN_JSON_PATH:-/workspace/ghy/HY-WorldPlay/datasets/preprocessed_gamefactory_f129_fixed/dataset_index.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-/workspace/ghy/HY-WorldPlay/outputs/dmd_init_pod_fixed}"

# ---- DMD 显存 / 训练帧数 / validation ----
export DMD_FSDP_CPU_OFFLOAD="${DMD_FSDP_CPU_OFFLOAD:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NUM_TRAINING_FRAMES="${NUM_TRAINING_FRAMES:-32}"
export MEMORY_FRAMES="${MEMORY_FRAMES:-20}"
export SLICE_LAST_FRAMES="${SLICE_LAST_FRAMES-}"
export VALIDATION_EVERY="${VALIDATION_EVERY:-100}"
export VALIDATION_SEED="${VALIDATION_SEED:-1}"

# ---- 训练步数 / checkpoint ----
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-700}"
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-100}"
export CHECKPOINTS_TOTAL_LIMIT="${CHECKPOINTS_TOTAL_LIMIT:-3}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# ---- 并行 (pod: 2x H200) ----
export NUM_GPUS="${NUM_GPUS:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MASTER_PORT="${MASTER_PORT:-29641}"

# ---- 可选 resume ----
if [ -n "${RESUME_FROM:-}" ]; then
  export RESUME_FROM_CHECKPOINT="$RESUME_FROM"
  echo "[run_dmd_train] resuming from $RESUME_FROM_CHECKPOINT"
fi

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/train.log}"

echo "[run_dmd_train] REPO_ROOT=$REPO_ROOT"
echo "[run_dmd_train] OUTPUT_DIR=$OUTPUT_DIR"
echo "[run_dmd_train] TRAIN_JSON_PATH=$TRAIN_JSON_PATH"
echo "[run_dmd_train] NUM_GPUS=$NUM_GPUS  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[run_dmd_train] MAX_TRAIN_STEPS=$MAX_TRAIN_STEPS CHECKPOINTING_STEPS=$CHECKPOINTING_STEPS CHECKPOINTS_TOTAL_LIMIT=$CHECKPOINTS_TOTAL_LIMIT"
echo "[run_dmd_train] LEARNING_RATE=$LEARNING_RATE MAX_GRAD_NORM=$MAX_GRAD_NORM CRITIC_LR=${CRITIC_LR:-<auto>}"
if [ -n "$SLICE_LAST_FRAMES" ]; then
  echo "[run_dmd_train] NUM_TRAINING_FRAMES=$NUM_TRAINING_FRAMES SLICE_LAST_FRAMES=$SLICE_LAST_FRAMES"
else
  echo "[run_dmd_train] NUM_TRAINING_FRAMES=$NUM_TRAINING_FRAMES SLICE_LAST_FRAMES=<unset/all-grad>"
fi
echo "[run_dmd_train] LOG_FILE=$LOG_FILE"

bash scripts/training/hyvideo15/run_ar_hunyuan_dmd_init.sh 2>&1 | tee "$LOG_FILE"
