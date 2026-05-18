# WorldPlay Causal Student OPD Plan

This note sketches how to adapt `/workspace/WorldPolicy/HY-WorldPlay` from the current causal student training path into an AnyFlow-style flow-map + on-policy distillation path, while preserving WorldPlay's long-video autoregressive/memory behavior.

Current reference points:

- WorldPlay training entry: `scripts/training/hyvideo15/run_ar_hunyuan_action_mem.sh`
- Current training doc: `trainer/README.md`
- Current AR/memory pipeline: `trainer/training/ar_hunyuan_mem_training_pipeline.py`
- Current AR transformer: `trainer/models/hyvideo/models/transformers/ar_action_hunyuanvideo_1_5_transformer.py`
- AnyFlow reference: `/workspace/WorldPolicy/AnyFlow/AnyFlow_profile.md`

We only focus on the **causal student** path. Bidirectional AnyFlow is useful as teacher/scoring reference, but it is not the target runtime model.

## Goal

Convert WorldPlay's autoregressive distilled student into a flow-map causal student that:

1. Keeps WorldPlay's long-video generation behavior:
   - rolling chunk generation;
   - reconstituted/context memory;
   - action-conditioned camera/control trajectory;
   - support for 30s+ and eventually longer rollouts.

2. Gains AnyFlow-style any-step/few-step capability:
   - train transition maps `z_t -> z_r`, not only instantaneous velocity/noise prediction;
   - use 4-step inference as the first target setting;
   - keep the door open for 2/4/8/16/... step scaling.

3. Can be OPD-refined on student self-rollouts:
   - reduce few-step discretization error;
   - reduce causal exposure bias over long-horizon rollout;
   - avoid optimizing only short clean clips.

## High-Level Two-Stage Plan

### Stage 1: Replace Ordinary Causal Student Pretraining With Flow-Map Causal Student Training

Current `trainer/README.md` describes ordinary autoregressive training with action control and memory design based on HunyuanVideo 1.5. The first documentation/code shift should be:

> ordinary causal student pretraining -> flow-map training for causal student.

The important constraint is that AnyFlow's released configs are mostly 81-frame/short-video centered, while WorldPlay's causal student must remain a long-video model. We should borrow AnyFlow's **two-time flow map formulation**, but not inherit its short-video training assumption as a hard limit.

#### 1.1 Add two-time conditioning to the WorldPlay AR transformer

AnyFlow adds a second time input `r_timestep` and trains the model to predict a transition from `t` to `r`. For WorldPlay, the AR transformer should accept:

- `timestep`: current noisy time `t`;
- `r_timestep`: target time `r`;
- optionally `clean_timestep`: for clean context/memory tokens;
- existing WorldPlay inputs:
  - image condition;
  - action/control condition;
  - text embeddings;
  - ByT5/glyph/vision states;
  - memory/context frames.

In AnyFlow, the model uses an interpolated time embedding:

```text
g * emb(t) + (1 - g) * emb'(r)
```

with `g = 0.25`. For WorldPlay, we can start with the same idea:

- initialize `emb'(r)` from the pretrained timestep embedder;
- keep a fixed gate, initially `0.25`;
- add `r_timestep` through the same conditioning path used by the HunyuanVideo transformer time embedding.

Candidate implementation area:

- `trainer/models/hyvideo/models/transformers/ar_action_hunyuanvideo_1_5_transformer.py`

There is already `use_meanflow` in the model signature/code path, so we should first check whether existing code has a partial MeanFlow/flow-map hook before adding a parallel implementation.

Current implementation status:

- Added a separate Stage-1 flow-map training entry so the original causal student training path remains intact:
  - `trainer/training/ar_hunyuan_flowmap_training_pipeline.py`
  - `scripts/training/hyvideo15/run_ar_hunyuan_action_mem_flowmap.sh`
- Reused the original WorldPlay AR/memory dataset, optimizer, checkpointing, and action/i2v conditioning.
- Added `flowmap_*` CLI flags to `TrainingArgs`.
- Initialized the additional `time_r_in` r-timestep embedder only when `--flowmap_training` is enabled, before FSDP wrapping, from the pretrained `time_in` weights.
- Updated the AR transformer forward path to:
  - sequence-parallel split `timestep_r` consistently with `timestep`;
  - apply `flowmap_gate_value * time_r_in(timestep_r)`;
  - error clearly if `timestep_r` is provided without a flow-map time embedder.

Smoke result:

```text
dataset: /workspace/WorldPolicy/HY-WorldPlay/datasets/preprocessed_gamefactory_sample10_f129/dataset_index.json
output:  /workspace/WorldPolicy/HY-WorldPlay/local_models/flowmap_causal_student_gamefactory_smoke_20260518_013043
steps:   1
loss:    2.03164
```

Checkpoint:

```text
/workspace/WorldPolicy/HY-WorldPlay/local_models/flowmap_causal_student_gamefactory_smoke_20260518_013043/checkpoint-1/transformer/diffusion_pytorch_model.safetensors
```

#### 1.2 Change the training target from instantaneous velocity to flow-map transition

Current causal training likely predicts the standard flow/velocity target for each AR training window. Stage 1 should switch to AnyFlow-style sampled `(t, r)`:

```text
t = max(rand(), rand())
r = min(rand(), rand())
z_t = (1 - t) * z_0 + t * eps
model_output = f_theta(z_t, t, r, context, action, prompt)
```

Target should follow the flow-map differential derivation equation:

```text
v = eps - z_0
dF_dt ~= [f(z_t + eps_fd * v, t + eps_fd, r)
          - f(z_t - eps_fd * v, t - eps_fd, r)] / (2 * eps_fd * guidance)
target = v - (t - r) * dF_dt
```

This is copied conceptually from AnyFlow's `train_causal()`.

Initial ratios can mirror AnyFlow:

```yaml
flowmap_cfg:
  gate_value: 0.25
  deltatime_type: r
  diffusion_ratio: 0.5
  consistency_ratio: 0.25
  epsilon: 5
```

Interpretation:

- some samples use `r = t`, becoming ordinary diffusion/flow matching;
- some samples use `r = 0`, becoming endpoint-style consistency;
- remaining samples use general `t > r`, becoming true flow-map transitions.

#### 1.3 Preserve WorldPlay long-video training

This is the main divergence from AnyFlow.

AnyFlow's public configs use 81-frame training and a fixed FAR chunk partition. WorldPlay must keep:

- long-horizon autoregressive memory;
- previous chunk/context reconstruction;
- action-conditioned transition across chunks;
- training windows that represent long rollouts, not just isolated short clips.

The Stage-1 flow-map training should therefore be inserted into WorldPlay's existing AR memory pipeline rather than replacing it with AnyFlow's 81-frame loop.

Proposed training unit:

```text
long video sample
  -> choose a current training window/chunk
  -> build previous context/memory from earlier chunks
  -> apply flow-map noising only to the current predicted chunk/window
  -> keep previous context/memory as clean or appropriately timed context
  -> predict z_t -> z_r for the current chunk conditioned on:
       previous memory/context,
       action trajectory,
       image/text conditions
```

Important design choice:

- Context frames from the past should usually remain clean with timestep `0`, because at inference they are generated/decoded past memory rather than the current noisy state.
- The current chunk/window receives `(t, r)`.
- If we later want robustness to noisy/self-generated history, we can add context corruption or self-context replay, but Stage 1 should first preserve the original WorldPlay training semantics.

#### 1.4 Update `trainer/README.md`

The README should eventually describe this as:

- "Flow-map causal student training with action and memory";
- dataset remains latent-preprocessed;
- add new arguments:
  - `--flowmap_training`;
  - `--flowmap_gate_value`;
  - `--flowmap_deltatime_type`;
  - `--flowmap_diffusion_ratio`;
  - `--flowmap_consistency_ratio`;
  - `--flowmap_epsilon`;
  - possibly `--opd_stage` later.

The old command can remain as "ordinary AR baseline", but the OPD branch should point to a new script, for example:

```bash
bash scripts/training/hyvideo15/run_ar_hunyuan_action_mem_flowmap.sh
```

Smoke command used:

```bash
cd /workspace/WorldPolicy/HY-WorldPlay
TRAIN_JSON_PATH=/workspace/WorldPolicy/HY-WorldPlay/datasets/preprocessed_gamefactory_sample10_f129/dataset_index.json \
OUTPUT_DIR=/workspace/WorldPolicy/HY-WorldPlay/local_models/flowmap_causal_student_gamefactory_smoke_$(date -u +%Y%m%d_%H%M%S) \
MAX_TRAIN_STEPS=1 \
CHECKPOINTING_STEPS=1 \
NUM_GPUS=2 \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/training/hyvideo15/run_ar_hunyuan_action_mem_flowmap.sh
```

### 1.5 Implementation audit: large `grad_norm` and current issues

The current flow-map smoke proves that the new forward/backward/checkpoint path can run, but it does **not** prove that the model parameters were updated.

Evidence from the two smoke runs:

```text
ordinary causal smoke:
  log:       local_models/causal_student_gamefactory_smoke_20260518_002904.log
  loss:      1.88733
  grad_norm: 8.94e17

flow-map causal smoke:
  log:       local_models/flowmap_causal_student_gamefactory_smoke_20260518_013043.log
  loss:      2.03164
  grad_norm: 2.14e18
```

So the huge `grad_norm` is not introduced only by the flow-map branch. The original causal student path already has the same failure mode on the GameFactory smoke dataset.

The key behavior is in `trainer/training/ar_hunyuan_mem_training_pipeline.py`:

```python
training_batch = self._clip_grad_norm(training_batch)
dist.all_reduce(grad_norm, op=dist.ReduceOp.MAX)

if self.global_rank == 0 and training_batch.grad_norm >= 10.0:
    print(self.global_rank, training_batch.grad_norm, training_batch.current_timestep, training_batch.video_path)

if training_batch.grad_norm < 10.0 or (not self.action):
    self.optimizer.step()
    self.lr_scheduler.step()
```

Important interpretation:

- `clip_grad_norm_while_handling_failing_dtensor_cases()` returns the **pre-clipping total norm**, matching PyTorch `clip_grad_norm_` semantics.
- The gradients may be clipped internally, but the action-training gate uses the returned pre-clipping value.
- With `action=True`, any `grad_norm >= 10` skips both `optimizer.step()` and `lr_scheduler.step()`.
- Therefore both smoke checkpoints above are best understood as "forward/backward/checkpoint succeeded"; they are not evidence of a real training update.

This also means that before running a longer Stage-1 flow-map training job, we should add an explicit metric such as `optimizer_step_skipped` or `did_optimizer_step`. Otherwise a run can appear healthy in loss/checkpoint logs while silently doing zero optimizer updates.

#### Flow-map-specific issues found in the first implementation

1. **Time sampling granularity differs from both WorldPlay and AnyFlow**

   AnyFlow samples one `(t, r)` per batch item and repeats it over frames. Original WorldPlay causal training samples one timestep per 4-latent chunk and repeats within that chunk. The current flow-map branch samples `batch_size * latent_t` independent times, effectively one `(t, r)` per latent frame.

   This may make the AR/memory objective noisier than intended and can break the local temporal semantics of WorldPlay's chunked causal training. The next version should sample `(t, r)` at WorldPlay's chunk granularity, then repeat across the corresponding latent frames.

2. **Outside-window memory training is only partially preserved**

   Original WorldPlay has special handling for `select_window_out_flag == 1`: previous chunks receive high/random timestep indices, while the loss is applied only to the last chunk. The flow-map branch preserves the last-chunk loss mask, but it does not yet reproduce the original previous-chunk timestep handling.

   For long-video training this matters, because those previous chunks are exactly the memory/context path we want to keep. The flow-map path should port the original outside-window timestep logic instead of only masking the loss.

3. **Two-time embedding is currently additive, not the AnyFlow-style replacement/interpolation**

   AnyFlow replaces the original condition embedder with a two-time embedder initialized from the original time embedder. The current WorldPlay change keeps the original `vec = time_in(t)` and then adds:

   ```python
   vec = vec + flowmap_gate_value * time_r_in(timestep_r)
   ```

   This changes the conditioning vector scale. Because `time_r_in` is initialized from `time_in`, the first flow-map forward effectively sees `emb(t) + 0.25 * emb(r)` instead of a normalized/interpolated two-time conditioning. This is a plausible contributor to unstable gradient scale, though the original causal path's huge norm shows it is not the only cause.

   Candidate fix: follow AnyFlow more closely by using a bounded two-time mixture, for example `g * emb(t) + (1 - g) * emb_r(r)` or the exact convention used by the AnyFlow `WanTwoTimeTextImageEmbedding` after we inspect that module in detail.

4. **Flow-map target scale needs one more verification pass**

   AnyFlow uses:

   ```python
   dF_dt = (pred_plus - pred_minus) / (2 * epsilon * guidance)
   target = (noise - latents) - (t - r) * dF_dt
   ```

   The current WorldPlay branch uses the same finite-difference perturbation size `epsilon / num_train_timesteps`, but then computes:

   ```python
   target = v_pred - ((timestep - r_timestep) / num_train_timesteps) * dF_dt
   ```

   This is a normalized-time convention. It may be mathematically defensible if we define `dF_dt` in scheduler-index units, but it is not a literal AnyFlow port. Before real training we should run an A/B smoke with the literal AnyFlow scale and log target norm/model output norm to decide which convention matches the pretrained model's scale.

5. **AnyFlow uses train weighting; current branch uses WorldPlay's unweighted masked MSE**

   AnyFlow applies `scheduler.get_train_weight(t)` and also rescales non-diffusion samples by the global diffusion loss. The current WorldPlay branch reuses the original WorldPlay masked MSE. That may be acceptable for keeping WorldPlay semantics, but it means our current flow-map loss is not a full AnyFlow reproduction.

   For the next training attempt, log per-sample/per-chunk loss by flow-map type: diffusion (`r=t`), consistency (`r=0`), and general transition (`t>r`). If the consistency/general buckets dominate the gradient, use AnyFlow's weighting or a warmup schedule.

6. **The GameFactory smoke dataset may amplify the inherited grad gate**

   The sample-10 GameFactory preprocessing is enough for plumbing, but it uses a tiny subset and generated metadata. Since original causal training already logs `8.94e17`, the immediate blocker is not specifically flow-map math; it is the combination of this data/config/model path with the original hard grad gate.

#### Priority before real Stage-1 training

<details>
<summary>Recommended order</summary>

- ✅ Add explicit logging for whether `optimizer.step()` actually ran.
- ✅ Run one ordinary causal smoke and one flow-map smoke with that metric, so we no longer infer step-skips indirectly from `grad_norm`.
- ✅ Fix flow-map timestep sampling to WorldPlay chunk granularity.
- ✅ Port the original outside-window timestep handling into the flow-map path.
- ✅ Change two-time embedding from additive scaling to an AnyFlow-compatible bounded mixture.
- ✅ Add debug logs for `model_pred`, `target`, `dF_dt`, `t-r`, and loss by flow-map sample type.
- ✅ Make the action grad-skip threshold configurable and add a debug bypass.
- ⬜ Run a multi-step smoke and confirm at least one real optimizer update occurs.

</details>

### 1.6 Stage-1 v1.1 implementation update

The next implementation pass addressed the first four concrete blockers above while keeping the original ordinary causal student training entry intact.

Changed files:

- `trainer/pipelines/pipeline_batch_info.py`
- `trainer/training/ar_hunyuan_mem_training_pipeline.py`
- `trainer/training/ar_hunyuan_flowmap_training_pipeline.py`
- `trainer/models/hyvideo/models/transformers/ar_action_hunyuanvideo_1_5_transformer.py`

#### Optimizer-step visibility

Training batches now carry:

```python
did_optimizer_step: bool
optimizer_step_skipped: bool
```

The shared AR training loop logs these values to the progress bar and wandb. When `grad_norm >= 10` under `action=True`, the printed diagnostic now explicitly includes `optimizer_step_skipped True`. This is intentionally implemented in the shared base pipeline, so both ordinary causal training and flow-map training expose the same signal.

#### Chunk-level `(t, r)` sampling

The flow-map branch no longer samples an independent `(t, r)` for every latent frame. It now samples at WorldPlay's 4-latent chunk granularity and repeats the sampled timestep across each chunk:

```text
chunk timestep sample -> repeat over 4 latent frames -> flatten to transformer timestep vector
```

This better matches the original WorldPlay causal training path, where timestep noise is piecewise constant over 4-latent chunks.

#### Outside-window timestep handling

For `select_window_out_flag == 1`, the flow-map path now mirrors the original causal memory-training behavior more closely:

- previous chunks before the final 4-latent current chunk receive random high timesteps sampled from scheduler indices `[500, 985)`;
- both `timestep` and `r_timestep` are forced to the same high value for those previous chunks;
- the loss mask still applies only to the final 4-latent chunk.

This keeps the previous-context/memory branch closer to the original long-video causal student training semantics.

#### AnyFlow-style bounded two-time embedding

The first flow-map implementation used:

```python
vec = emb(t) + gate * emb_r(r)
```

After checking AnyFlow's `WanTwoTimeTextImageEmbedding`, the WorldPlay transformer now uses the bounded mixture:

```python
vec = (1 - gate) * emb(t) + gate * emb_r(delta)
```

where `delta` is either:

- `r` when `flowmap_deltatime_type == "r"`;
- `t - r` when `flowmap_deltatime_type == "t-r"`.

The default remains `gate = 0.25` and `deltatime_type = "r"`, matching the current flow-map config.

#### v1.1 smoke result

Command shape:

```bash
cd /workspace/WorldPolicy/HY-WorldPlay
TRAIN_JSON_PATH=/workspace/WorldPolicy/HY-WorldPlay/datasets/preprocessed_gamefactory_sample10_f129/dataset_index.json \
OUTPUT_DIR=/workspace/WorldPolicy/HY-WorldPlay/local_models/flowmap_causal_student_gamefactory_v2_smoke_20260518_034436 \
MAX_TRAIN_STEPS=1 \
CHECKPOINTING_STEPS=1 \
NUM_GPUS=2 \
CUDA_VISIBLE_DEVICES=0,1 \
WANDB_MODE=offline \
bash scripts/training/hyvideo15/run_ar_hunyuan_action_mem_flowmap.sh
```

Result:

```text
output:                  local_models/flowmap_causal_student_gamefactory_v2_smoke_20260518_034436
checkpoint:              checkpoint-1/transformer/diffusion_pytorch_model.safetensors
loss:                    1.72025
step_time:               48.58s
grad_norm:               2.3157e18
did_optimizer_step:      0
optimizer_step_skipped:  1
```

Interpretation:

- The v1.1 implementation runs through forward, backward, explicit step-skip logging, and checkpointing.
- The loss is lower than the first flow-map smoke (`2.03164 -> 1.72025`), but this is only a one-sample smoke and should not be overinterpreted.
- The important confirmed blocker is now explicit: the shared WorldPlay action-training grad gate still skips the optimizer update because the pre-clipping `grad_norm` is far above `10`.
- The saved v1.1 checkpoint is therefore still a plumbing checkpoint, not proof of a parameter-updated flow-map student.

Next immediate debugging target:

1. Log norm scale before the loss backward path: `model_pred`, `target`, `v_pred`, `dF_dt`, and `t-r`. Done in v1.2.
2. Compare original causal target norm and flow-map target norm on the same batch. Partially done for the diffusion bucket, where `target == v_pred`.
3. Decide whether to relax/parameterize the `grad_norm < 10` action gate for smoke/debug runs, or to fix the upstream scale first.
4. After at least one real `did_optimizer_step=1` smoke, run a short multi-step training job.

### 1.7 Stage-1 v1.2 norm debugging

The v1.2 pass adds flow-map scale diagnostics directly inside `_transformer_forward_and_compute_loss()` before `loss.backward()`.

The logged metrics include:

- `flowmap_debug/model_pred_abs_mean`, `rms`, `abs_max`
- `flowmap_debug/target_abs_mean`, `rms`, `abs_max`
- `flowmap_debug/v_pred_abs_mean`, `rms`, `abs_max`
- `flowmap_debug/dF_dt_abs_mean`, `rms`, `abs_max`
- `flowmap_debug/t_minus_r_abs_mean`, `rms`, `abs_max`
- masked versions of model/target statistics
- `flowmap_debug/diff_abs_mean`
- `flowmap_debug/timestep_abs_mean`
- `flowmap_debug/r_timestep_abs_mean`

These are stored in `training_batch.flowmap_debug_metrics`, printed on rank 0, and sent to wandb from the shared training loop.

#### v1.2 smoke result

Command shape:

```bash
cd /workspace/WorldPolicy/HY-WorldPlay
TRAIN_JSON_PATH=/workspace/WorldPolicy/HY-WorldPlay/datasets/preprocessed_gamefactory_sample10_f129/dataset_index.json \
OUTPUT_DIR=/workspace/WorldPolicy/HY-WorldPlay/local_models/flowmap_causal_student_gamefactory_debugnorm_smoke_20260518_035817 \
MAX_TRAIN_STEPS=1 \
CHECKPOINTING_STEPS=1 \
NUM_GPUS=2 \
CUDA_VISIBLE_DEVICES=0,1 \
WANDB_MODE=offline \
bash scripts/training/hyvideo15/run_ar_hunyuan_action_mem_flowmap.sh
```

Result:

```text
output:                  local_models/flowmap_causal_student_gamefactory_debugnorm_smoke_20260518_035817
checkpoint:              checkpoint-1/transformer/diffusion_pytorch_model.safetensors
loss:                    1.7203
step_time:               51.24s
grad_norm:               2.3419e18
did_optimizer_step:      0
optimizer_step_skipped:  1
```

Key debug metrics:

```text
model_pred_rms:          1.0045
target_rms:              1.6496
v_pred_rms:              1.6496
dF_dt_rms:               0.0361
t_minus_r_rms:           0.0000
masked_model_pred_rms:   0.2483
masked_target_rms:       0.6876
diff_abs_mean:           0.2867
timestep_abs_mean:       249.5
r_timestep_abs_mean:     249.5
```

Interpretation:

- This smoke landed in the diffusion bucket (`t-r = 0`), so the flow-map target reduced to the ordinary velocity target: `target == v_pred`.
- The target/model scales are not exploding: `target_rms ~= 1.65`, `model_pred_rms ~= 1.00`, and `dF_dt_rms ~= 0.036`.
- Therefore this particular huge `grad_norm` is unlikely to be caused by an obviously exploding flow-map target.
- Because the original ordinary causal smoke also produced `grad_norm ~= 8.94e17`, the current evidence points more strongly at the inherited action-training grad gate / FSDP norm accounting / full-model Muon update scale than at flow-map target scale.
- We still need a non-diffusion bucket sample (`t-r > 0`) to validate the general flow-map target term. For deterministic debugging, temporarily setting `flowmap_diffusion_ratio=0` would force non-diffusion samples.

Recommended next action:

1. Add a smoke/debug-only CLI switch to bypass or parameterize the `grad_norm < 10` action gate. Done in v2.
2. Run one debug smoke with the gate disabled after clipping to confirm parameters can update without NaNs. Done in v2.
3. Run one non-diffusion flow-map smoke (`diffusion_ratio=0`, likely `consistency_ratio=0` or `0.25`) and compare target/dF_dt scale.
4. If update is numerically stable, keep the gate configurable for research runs instead of hard-coded. Done in v2.

### 1.8 Stage-1 v2 optimizer-step fix

The root issue was not that the flow-map target scale was obviously exploding. The immediate blocker was the inherited hard-coded action-training optimizer-step gate:

```python
if grad_norm < 10.0 or not action:
    optimizer.step()
```

Because `clip_grad_norm_` returns the pre-clipping total norm, this gate skipped the update even after gradients were clipped. The fix keeps the original default behavior, but makes it configurable and debuggable.

New training args:

```text
--action_grad_skip_threshold 10.0
--debug_disable_action_grad_skip False
```

The flow-map launch script exposes these via env vars:

```bash
ACTION_GRAD_SKIP_THRESHOLD=10.0
DEBUG_DISABLE_ACTION_GRAD_SKIP=False
```

For debug smoke runs, use:

```bash
DEBUG_DISABLE_ACTION_GRAD_SKIP=True
```

This means:

- default training still preserves the original WorldPlay safety gate;
- smoke/debug runs can intentionally step after gradient clipping;
- `did_optimizer_step` and `optimizer_step_skipped` make the behavior explicit.

#### v2 smoke result

Command shape:

```bash
cd /workspace/WorldPolicy/HY-WorldPlay
TRAIN_JSON_PATH=/workspace/WorldPolicy/HY-WorldPlay/datasets/preprocessed_gamefactory_sample10_f129/dataset_index.json \
OUTPUT_DIR=/workspace/WorldPolicy/HY-WorldPlay/local_models/flowmap_causal_student_gamefactory_stepenabled_smoke_20260518_041350 \
MAX_TRAIN_STEPS=1 \
CHECKPOINTING_STEPS=1 \
NUM_GPUS=2 \
CUDA_VISIBLE_DEVICES=0,1 \
WANDB_MODE=offline \
DEBUG_DISABLE_ACTION_GRAD_SKIP=True \
bash scripts/training/hyvideo15/run_ar_hunyuan_action_mem_flowmap.sh
```

Result:

```text
output:                  local_models/flowmap_causal_student_gamefactory_stepenabled_smoke_20260518_041350
checkpoint:              checkpoint-1/transformer/diffusion_pytorch_model.safetensors
loss:                    1.7203
step_time:               56.41s
grad_norm:               2.2e18
did_optimizer_step:      1
optimizer_step_skipped:  0
```

The run completed forward, backward, gradient clipping, Muon optimizer step, LR scheduler step, distributed checkpoint save, and consolidated checkpoint save without NaNs or runtime failure.

Working conclusion:

- The immediate "no real training update" blocker is fixed for debug/smoke runs.
- The huge pre-clipping `grad_norm` is still present and should remain visible, but it no longer silently prevents a deliberately enabled debug update.
- Before a longer real run, keep `did_optimizer_step` in the logs and decide whether production flow-map training should use the original `10.0` threshold, a higher threshold, or staged gate disabling.
- Still run a non-diffusion sample smoke to inspect the true flow-map transition term where `t-r > 0`.

Summary phrase:

```text
v2: configurable grad-gate debug step, 4-latent flow-map timestep, bounded two-time embedding
```

## Stage 2: OPD on the Flow-Map Causal Student

After Stage 1 converges, we run on-policy flow-map distillation on the causal student.

First target: **4-step OPD**, because WorldPlay's current released `ar_distilled_action_model` is already positioned as a 4-step distilled AR model.

But the distilled student must still support long video generation. OPD cannot be restricted to one isolated 5s clip if our target behavior is 30s+ interactive rollout.

### 2.1 Student in OPD

The OPD student is the Stage-1 flow-map causal WorldPlay model.

It should self-rollout under the same generation logic used at inference:

```text
prompt/image/action
  -> generate previous trajectory chunks
  -> maintain/reconstitute memory/context
  -> generate current chunk with 4 flow-map steps
```

Example training rollout:

```text
student rolls 30s previous trajectory
student rolls 5s current trajectory
OPD loss is applied to current 5s, with previous 30s treated as self-generated context
```

This is the causal exposure-bias setup we care about. The student sees its own imperfect long-history context, not only clean teacher-forced history.

### 2.2 Teacher in OPD

Initial/simple version: follow AnyFlow's OPD structure.

Roles:

- `student`: Stage-1 WorldPlay flow-map causal model being updated.
- `real_score` / teacher: strong WorldPlay teacher model used to score/re-noise current chunk distribution.
- `discriminator` / fake score: initialized from teacher or a teacher-like checkpoint, then trained on student-generated samples.

Candidate teacher choices, in increasing sophistication:

1. Existing WorldPlay bidirectional model as teacher score.
2. Existing WorldPlay AR non-distilled model as teacher score.
3. A future WorldPlay original distillation teacher once we inspect/recover the original distill code.

For the first OPD design doc, we can keep it AnyFlow-like:

```text
real_score = teacher checkpoint, eval/frozen
discriminator = teacher-initialized fake score model, trainable
student = flow-map causal student, trainable
```

The DMD gradient is:

```text
grad ~= pred_fake_current_chunk - pred_real_current_chunk
student loss: current_chunk -> current_chunk - grad
```

### 2.3 Context mismatch issue

This is the biggest WorldPlay-specific problem.

In the long-video setup:

```text
student context: generated 0-30s trajectory + generated 31-35s current chunk
teacher context: may only see / score the 31-35s current chunk context
```

So teacher and student are not naturally conditioned on the same history.

For the **first implementation**, we can follow AnyFlow's simpler assumption:

- use the student self-rollout to produce `z_0` current chunk;
- re-noise the current chunk;
- compute teacher/fake scores on that current chunk with the available conditioning;
- backprop through the student's flow-map backward simulation chain.

This means teacher may only score the local current window under a simplified context. It will not perfectly judge 30s global consistency yet.

This is acceptable for the first OPD version because:

- it matches AnyFlow's baseline OPD structure;
- it gives us a working 4-step OPD training loop;
- later we can refine context alignment when we have the original WorldPlay distillation code.

### 2.4 Later refinement: align teacher/student context

After the first AnyFlow-style OPD works, we should refine context alignment.

Possible directions:

1. **Teacher sees reconstructed student history**
   - Feed teacher the same 0-30s student-generated memory/context used by the student.
   - Hard if teacher is bidirectional and expects short/full windows.

2. **Teacher scores only current chunk but with memory embeddings**
   - Convert student history into memory states compatible with teacher.
   - This is closest to preserving WorldPlay's reconstituted context memory.

3. **Teacher-forced prefix, student current chunk**
   - Use clean/teacher prefix for both teacher and student context during early OPD.
   - Then gradually replace prefix with student-generated context.
   - This is a curriculum to avoid unstable early rollouts.

4. **Hybrid local/global loss**
   - DMD on current 5s chunk;
   - additional action-following/geometric consistency reward/loss over longer rollout;
   - possibly reuse WorldCompass ideas later.

### 2.5 Flow-map backward simulation for WorldPlay chunks

For 4-step current-chunk OPD, the flow-map chain mirrors AnyFlow:

```text
sample_step = 4
pick grad_timestep k in [0, 3]
T -> t_k      shortcut flow-map transition
t_k -> r_k    target transition
r_k -> 0      shortcut flow-map transition
```

But for WorldPlay we apply this inside a causal chunk rollout:

```text
for each current chunk:
  use previous generated chunks as memory/context
  run flow-map backward simulation for current chunk
  DMD re-noise current chunk
  update student from teacher/fake score gradient
```

Long video support comes from repeating the chunk rollout, not from making a single short 81-frame OPD sample.

### 2.6 OPD validation

Validation should include both short and long generation:

- 5s / 125-frame default example;
- 30s / 717-frame example, like `ar_distilled_castle_717`;
- mixed action trajectories, not only `w-*`;
- metrics:
  - visual quality;
  - action following;
  - temporal/geometric consistency;
  - context drift over time;
  - performance at 2/4/8/16 steps if any-step behavior is desired.

## Proposed Implementation Milestones

### Milestone A: Documentation + config skeleton

- Update `trainer/README.md` to describe flow-map causal student training.
- Add a training script skeleton:
  - `scripts/training/hyvideo15/run_ar_hunyuan_action_mem_flowmap.sh`
- Add config/args for flow-map training.

### Milestone B: Flow-map forward training

- Add `r_timestep` support to AR transformer.
- Add two-time embedding path.
- Add flow-map target computation to AR memory training pipeline.
- Preserve long-video memory/context sampling.
- Train Stage-1 flow-map causal student.

### Milestone C: 4-step OPD prototype

- Add student rollout path that can generate:
  - previous long context;
  - current chunk under 4-step flow-map sampling.
- Add teacher/real score model loader.
- Add discriminator/fake score model loader.
- Implement DMD loss on current chunk.
- Start with AnyFlow-style local current-window teacher context.

### Milestone D: Context-aligned OPD refinement

- Use WorldPlay original distillation code once available.
- Align teacher and student context/memory.
- Add curriculum from clean context to student-generated context.
- Extend losses/metrics for long-horizon consistency.

## Key Risks

1. **Short-video bias from AnyFlow**
   - If we copy AnyFlow training too literally, the student may improve 4-step short clips but lose long-video stability.
   - Mitigation: keep WorldPlay's AR memory pipeline as the base loop.

2. **Teacher/student context mismatch**
   - Teacher may score only local current chunk while student errors are long-history errors.
   - Mitigation: start local, then refine context alignment.

3. **OPD memory cost**
   - Backprop through student rollout plus teacher/fake score is expensive.
   - Mitigation: first train only current chunk with truncated/shortcut flow-map chain; use sequence parallelism and memory recomputation.

4. **Action/control drift**
   - DMD alone may improve visual realism but weaken action control.
   - Mitigation: preserve action-conditioned training losses and add action-following validation.

5. **Any-step vs 4-step tension**
   - If OPD overfocuses on 4 steps, any-step behavior may degrade.
   - Mitigation: after 4-step prototype, sample rollout steps `[2, 4, 8, 16]` as AnyFlow does.

## Working Conclusion

The first clean design is:

```text
Stage 1:
  WorldPlay AR memory training
  -> replace current causal student target with flow-map causal target
  -> keep long-video context/memory training intact

Stage 2:
  load Stage-1 flow-map causal student
  -> run 4-step student self-rollout over long context + current chunk
  -> apply AnyFlow-style OPD/DMD on current chunk
  -> initially accept simplified teacher context
  -> later refine teacher/student context alignment using original WorldPlay distill code
```

This gives us an incremental path: first make WorldPlay's causal student flow-map capable, then add OPD without sacrificing the long-video behavior that makes WorldPlay different from AnyFlow.
