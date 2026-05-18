#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/WorldPolicy/HY-WorldPlay}"
cd "$REPO_ROOT"

if [ -f "$REPO_ROOT/activate_worldplay.sh" ]; then
  source "$REPO_ROOT/activate_worldplay.sh"
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false

MODEL_ROOT="${MODEL_ROOT:-$REPO_ROOT/local_models}"
TRANSFORMER_MODEL_PATH="${TRANSFORMER_MODEL_PATH:-$MODEL_ROOT/hunyuanvideo_1_5/transformer/480p_i2v}"
AR_DISTILL_ACTION_MODEL_PATH="${AR_DISTILL_ACTION_MODEL_PATH:-$MODEL_ROOT/ar_distilled_action_model/ar_distilled_action_model/model.safetensors}"
MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/hunyuanvideo_1_5}"
TRAIN_JSON_PATH="${TRAIN_JSON_PATH:-$REPO_ROOT/datasets/preprocessed_gamefactory_sample10_f129/dataset_index.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$MODEL_ROOT/flowmap_causal_student_train_$(date -u +%Y%m%d_%H%M%S)}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-200000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-500}"
WANDB_KEY="${WANDB_KEY:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-worldplay-flowmap-causal-student}"

NUM_GPUS="${NUM_GPUS:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

training_args=(
  --json_path "$TRAIN_JSON_PATH"
  --causal
  --action
  --i2v_rate 0.2
  --train_time_shift 3.0
  --window_frames 24
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
  --num_frames 77
  --enable_gradient_checkpointing_type "full"
  --seed 3208
  --weighting_scheme "logit_normal"
  --logit_mean 0.0
  --logit_std 1.0
  --flowmap_training
  --flowmap_gate_value 0.25
  --flowmap_deltatime_type "r"
  --flowmap_diffusion_ratio 0.5
  --flowmap_consistency_ratio 0.25
  --flowmap_epsilon 5.0
)

parallel_args=(
  --num_gpus $((NUM_GPUS * 1))
  --sp_size "$NUM_GPUS"
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim "$NUM_GPUS"
)

model_args=(
  --cls_name "HunyuanTransformer3DARActionModel"
  --load_from_dir "$TRANSFORMER_MODEL_PATH"
  --ar_action_load_from_dir "$AR_DISTILL_ACTION_MODEL_PATH"
  --model_path "$MODEL_PATH"
  --pretrained_model_name_or_path "$MODEL_PATH"
)

dataset_args=(
  --dataloader_num_workers 1
)

validation_args=(
  --validation_steps 200
  --validation_sampling_steps "50"
  --validation_guidance_scale "6.0"
)

optimizer_args=(
  --learning_rate 1e-5
  --mixed_precision "bf16"
  --checkpointing_steps "$CHECKPOINTING_STEPS"
  --weight_decay 1e-4
  --max_grad_norm 1.0
)

miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.1
  --multi_phased_distill_schedule "4000-1"
  --not_apply_cfg_solver
  --dit_precision "fp32"
  --num_euler_timesteps 50
  --ema_start_step 0
)

export MASTER_PORT="${MASTER_PORT:-29621}"

torchrun \
  --master_port="$MASTER_PORT" \
  --nproc_per_node="$NUM_GPUS" \
  --nnodes 1 \
  trainer/training/ar_hunyuan_flowmap_training_pipeline.py \
  "${parallel_args[@]}" \
  "${model_args[@]}" \
  "${dataset_args[@]}" \
  "${training_args[@]}" \
  "${optimizer_args[@]}" \
  "${validation_args[@]}" \
  "${miscellaneous_args[@]}"
