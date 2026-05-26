# WorldPlay DMD self-forcing distillation

DMD2 + self-forcing distillation of the HY-WorldPlay AR causal student
(`HY-World1.5-Autoregressive-480P-I2V`) into a 4-step distilled student.
Reference: LongLive (`NVlabs/LongLive`). KV cache uses HY-WorldPlay's existing
per-chunk `select_aligned_memory_frames` prefill — no sink / local-window,
no prompt-switch re-cache.

Entry point: `dmd_init_training_pipeline.py`
Launcher: `scripts/training/hyvideo15/run_ar_hunyuan_dmd_init.sh`

All DMD-specific knobs are passed as **environment variables** to the launcher
(the `TrainingArgs` dataclass drops unknown CLI flags, so env vars are used
instead).

## Gradient scope: all chunks vs only the last chunk

A self-forcing rollout generates `NUM_TRAINING_FRAMES` latents chunk by chunk
(`chunk_latent_frames = 4`, so 32 latents = 8 chunks). `SLICE_LAST_FRAMES`
controls which chunks carry gradient:

| `SLICE_LAST_FRAMES`        | behaviour                                              | LongLive analog |
|----------------------------|--------------------------------------------------------|-----------------|
| unset / empty (default)    | **all chunks** carry gradient                          | `train_init`    |
| `4`                        | **only the last chunk** carries gradient               | `train_long`    |
| `8`                        | only the last 2 chunks carry gradient                  | —               |
| `4*k`                      | only the last `k` chunks carry gradient                | —               |

### How to enable "only the last chunk has gradient"

Set `SLICE_LAST_FRAMES=4` when launching:

```bash
SLICE_LAST_FRAMES=4 \
DMD_FSDP_CPU_OFFLOAD=1 \
NUM_TRAINING_FRAMES=32 \
MEMORY_FRAMES=20 \
NUM_GPUS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/training/hyvideo15/run_ar_hunyuan_dmd_init.sh
```

The **unit is latent frames, not chunks**. One chunk = `chunk_latent_frames` = 4
latents. The mapping is `grad_chunks = max(1, SLICE_LAST_FRAMES // 4)`:

- `SLICE_LAST_FRAMES=4` → `4 // 4 = 1` chunk → only chunk 7 of 8 has gradient.
- `SLICE_LAST_FRAMES=1` → `1 // 4 = 0`, floored by `max(1, ...)` to 1 chunk —
  same result, but `=4` is the semantically clean value.

With only the last chunk active, chunks `0 .. N-2` run under `torch.no_grad()`
(they still execute the rollout to build context / KV cache, but store no
activations). Only the last chunk's denoise forward is differentiable, so the
backward activation footprint drops from `N` chunks to ~1 chunk. This is what
makes large `NUM_TRAINING_FRAMES` (e.g. 32) fit on 4 GPUs.

Trade-off: the generator only receives a DMD loss from the final chunk each
step; earlier chunks are pure context.

## Other environment variables

| env var                  | default | meaning |
|---------------------------|---------|---------|
| `NUM_TRAINING_FRAMES`     | `32`    | latents per self-forcing rollout (32 ≈ 5 s) |
| `MEMORY_FRAMES`           | `16`    | history latents prefilled into the per-chunk KV cache |
| `SLICE_LAST_FRAMES`       | unset   | trailing latents with gradient — see above |
| `DMD_FSDP_CPU_OFFLOAD`    | `1`     | offload FSDP params / grads / optimizer state to CPU |
| `DMD_PER_CHUNK_CHECKPOINT`| `0`     | wrap each grad chunk's denoise in `torch.utils.checkpoint` (recompute on backward) |
| `DFAKE_GEN_UPDATE_RATIO`  | `5`     | generator updated every N critic steps |
| `REAL_GUIDANCE_SCALE`     | `6.0`   | CFG scale for the frozen BI teacher (`real_score`) |
| `FAKE_GUIDANCE_SCALE`     | `0.0`   | CFG scale for the trainable BI critic (`fake_score`) |
| `REAL_SCORE_LOAD_FROM_DIR`| —       | BI bidirectional checkpoint path (required) |

## Memory notes (4×H200, 140 GB/GPU)

- All chunks grad, no cpu_offload: max `NUM_TRAINING_FRAMES = 20`.
- All chunks grad, `DMD_FSDP_CPU_OFFLOAD=1`: max `NUM_TRAINING_FRAMES = 24`.
- `NUM_TRAINING_FRAMES = 32` all chunks grad: needs either 8 GPUs (sequence
  parallelism halves per-GPU activations + KV) or `SLICE_LAST_FRAMES=4`.
- `MEMORY_FRAMES` only caps the prefill forward peak; it does **not** reduce the
  per-chunk gradient activation accumulation, so lowering it alone will not make
  a larger `NUM_TRAINING_FRAMES` fit.
