#!/usr/bin/env bash
# WorldPlay DMD self-forcing init-phase training launcher.
#
# Distills the 50-step AR causal student (HY-World1.5-Autoregressive-480P-I2V)
# into a 4-step distilled student using LongLive-style DMD + self-forcing
# rollout, but with WorldPlay's existing per-chunk memory-frame KV cache
# (no sink/local-window, no prompt-switch re-cache).
#
# Reference output ckpt (for QC after training):
#   local_models/ar_distilled_action_model/diffusion_pytorch_model.safetensors
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/raid/yiren/ghy/new/HY-WorldPlay}"
cd "$REPO_ROOT"

if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "worldplay" ]; then
  source /home/Yirenteam/miniconda3/etc/profile.d/conda.sh
  conda activate worldplay
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false

# NCCL flight-recorder: if a collective hangs (e.g. rank divergence under
# cpu_offload Muon patch), dump per-rank trace on watchdog timeout so we
# can see which param caused the divergence.
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-4096}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="${TORCH_NCCL_DEBUG_INFO_TEMP_FILE:-$REPO_ROOT/outputs/nccl_trace}"

HF_HUNYUAN="${HF_HUNYUAN:-/home/Yirenteam/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038}"
HF_HY="${HF_HY:-/home/Yirenteam/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/f4c29235647707b571479a69b569e4166f9f5bf8}"

MODEL_ROOT="${MODEL_ROOT:-$REPO_ROOT/local_models}"
TRANSFORMER_MODEL_PATH="${TRANSFORMER_MODEL_PATH:-$HF_HUNYUAN/transformer/480p_i2v}"
AR_ACTION_MODEL_PATH="${AR_ACTION_MODEL_PATH:-$HF_HY/ar_model/diffusion_pytorch_model.safetensors}"
BI_ACTION_MODEL_PATH="${BI_ACTION_MODEL_PATH:-$HF_HY/bidirectional_model/diffusion_pytorch_model.safetensors}"
MODEL_PATH="${MODEL_PATH:-$HF_HUNYUAN}"
TRAIN_JSON_PATH="${TRAIN_JSON_PATH:-$REPO_ROOT/datasets/preprocessed_gamefactory_f129/dataset_index.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/dmd_init_$(date -u +%Y%m%d_%H%M%S)}"

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-700}"        # LongLive train_init default
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-100}"
CHECKPOINTS_TOTAL_LIMIT="${CHECKPOINTS_TOTAL_LIMIT:-3}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
WANDB_KEY="${WANDB_KEY:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-worldplay-dmd-init}"

# DMD knobs (passed via env, NOT via CLI — TrainingArgs dataclass would drop them)
export NUM_TRAINING_FRAMES="${NUM_TRAINING_FRAMES:-32}"
export MEMORY_FRAMES="${MEMORY_FRAMES:-20}"
export DFAKE_GEN_UPDATE_RATIO="${DFAKE_GEN_UPDATE_RATIO:-5}"
export REAL_GUIDANCE_SCALE="${REAL_GUIDANCE_SCALE:-6.0}"
export FAKE_GUIDANCE_SCALE="${FAKE_GUIDANCE_SCALE:-0.0}"
export REAL_SCORE_LOAD_FROM_DIR="$BI_ACTION_MODEL_PATH"

NUM_GPUS="${NUM_GPUS:-1}"                          # smoke runs on 1 GPU
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

training_args=(
  --json_path "$TRAIN_JSON_PATH"
  --causal
  --action
  --i2v_rate 0.2
  --train_time_shift 3.0   # Match stage1 (was 5.0; inference 480p_i2v default is 5.0,
                           # so when running OUR distilled student must pass
                           # --flow_shift 3.0 to generate.py or update commons/__init__).
  --window_frames 32
  --wandb_key "$WANDB_KEY"
  --wandb_entity "$WANDB_ENTITY"
  --tracker_project_name "$WANDB_PROJECT"
  --output_dir "$OUTPUT_DIR"
  --max_train_steps "$MAX_TRAIN_STEPS"
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 9
  --num_height 480
  --num_width 832
  --num_frames 125
  --enable_gradient_checkpointing_type "full"
  --seed 3208
  --weighting_scheme "logit_normal"
  --logit_mean 0.0
  --logit_std 1.0
  --num-training-frames "$NUM_TRAINING_FRAMES"
  --dfake-gen-update-ratio "$DFAKE_GEN_UPDATE_RATIO"
  # DMD score-specific knobs are still read via env:
  # REAL_GUIDANCE_SCALE, FAKE_GUIDANCE_SCALE, REAL_SCORE_LOAD_FROM_DIR.
)

parallel_args=(
  --num_gpus $NUM_GPUS
  --sp_size $NUM_GPUS
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim $NUM_GPUS
)

model_args=(
  --cls_name "HunyuanTransformer3DARActionModel"
  --load_from_dir "$TRANSFORMER_MODEL_PATH"
  --ar_action_load_from_dir "$AR_ACTION_MODEL_PATH"
  --model_path "$MODEL_PATH"
  --pretrained_model_name_or_path "$MODEL_PATH"
)

dataset_args=(
  --dataloader_num_workers 0
)

validation_args=(
  --validation_steps 200
  --validation_sampling_steps "4"
  --validation_guidance_scale "6.0"
)

optimizer_args=(
  --learning_rate "$LEARNING_RATE"
  --mixed_precision "bf16"
  --checkpointing_steps "$CHECKPOINTING_STEPS"
  --weight_decay 1e-4
  --max_grad_norm "$MAX_GRAD_NORM"
)

miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit "$CHECKPOINTS_TOTAL_LIMIT"
  --training_cfg_rate 0.0
  --multi_phased_distill_schedule "4000-1"
  --not_apply_cfg_solver
  --dit_precision "fp32"
  --num_euler_timesteps 4
  --ema_start_step 0
)

export MASTER_PORT="${MASTER_PORT:-29641}"

torchrun \
  --master_port="$MASTER_PORT" \
  --nproc_per_node="$NUM_GPUS" \
  --nnodes 1 \
  trainer/training/dmd/dmd_init_training_pipeline.py \
  "${parallel_args[@]}" \
  "${model_args[@]}" \
  "${dataset_args[@]}" \
  "${training_args[@]}" \
  "${optimizer_args[@]}" \
  "${validation_args[@]}" \
  "${miscellaneous_args[@]}"
