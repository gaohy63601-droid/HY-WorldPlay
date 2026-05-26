# SPDX-License-Identifier: Apache-2.0
"""
DMD init-phase training pipeline for WorldPlay AR causal student.

This pipeline is the WorldPlay analog of LongLive `train_init.sh`:

  - one training step = one generator self-rollout (32 latents = 5s) under
    fresh KV cache, plus one critic self-rollout (also fresh KV cache).
  - generator is updated every `dfake_gen_update_ratio` steps (default 5),
    critic is updated every step.
  - DMD loss formula follows LongLive `model/dmd.py`.
  - Self-forcing rollout follows LongLive but uses WorldPlay's existing
    per-chunk `select_aligned_memory_frames` prefill (`see worldplay_video_pipeline._ar_rollout_inner`).

Three transformers (generator, real_score, fake_score) all share the
`DMDHunyuanTransformer` class. They differ only by which safetensors they load:

    generator   <- AR_ACTION_MODEL_PATH    (causal student start point)
    real_score  <- BI_ACTION_MODEL_PATH    (bidirectional teacher; frozen)
    fake_score  <- BI_ACTION_MODEL_PATH    (bidirectional critic; trainable)

No existing files are modified. All registration / wiring is done in this file
and in the launch script.
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import wandb
from safetensors.torch import load_file

sys.path.append(os.path.abspath("."))

from trainer.distributed import get_local_torch_device, get_sp_group, get_world_group
from trainer.distributed.parallel_state import get_sp_parallel_rank, get_sp_world_size
from trainer.logger import init_logger
from trainer.pipelines import TrainingBatch
from trainer.training.ar_hunyuan_mem_training_pipeline import TrainingPipeline
from trainer.training.activation_checkpoint import apply_activation_checkpointing
from trainer.training.training_utils import (
    clip_grad_norm_while_handling_failing_dtensor_cases,
    load_checkpoint,
    save_checkpoint,
)
from trainer.training.muon import get_muon_optimizer
from trainer.trainer_args import TrainerArgs, TrainingArgs
from trainer.utils import FlexibleArgumentParser, set_random_seed

from trainer.training.dmd.dmd_model import WorldPlayDMD
from trainer.training.dmd.dmd_transformer import DMDHunyuanTransformer
from trainer.training.dmd.self_forcing_pipeline import WorldPlaySelfForcingPipeline


def _patch_muon_for_cpu_offload() -> None:
    """Make `trainer.training.muon.Muon.step` survive FSDP cpu_offload.

    Two problems we have to work around:

      1. `zeropower_via_newtonschulz5(G)` does `G.full_tensor()` which is an
         all-gather over the mesh. With FSDP cpu_offload, the DTensor local
         lives on CPU but the mesh is CUDA; NCCL won't all-gather CPU tensors
         ("no backend type for cpu"). Fix: lift the local to CUDA before
         all-gather, run NS5 on CUDA, then bring the result back to CPU.

      2. The post-NS5 update `p.data.add_(u.view(p.shape), alpha=-lr)` goes
         through DTensor dispatch. Even though our patched ns5 returns a
         CPU-local DTensor, the dispatcher's local op call ends up mixing a
         cuda:N tensor with a CPU one (PyTorch 2.6 DTensor doesn't fully
         support cpu-local + cuda-mesh combo through pointwise dispatch).
         Fix: bypass DTensor entirely for params whose local is on CPU and
         do the in-place update directly on `_local_tensor`.
    """
    import math as _math
    from trainer.training import muon as _muon_mod
    from torch.distributed.tensor import DTensor as _DTensor

    if getattr(_muon_mod, "_cpuoff_patched", False):
        return

    _orig_ns5 = _muon_mod.zeropower_via_newtonschulz5

    def _patched_ns5(G, steps: int = 5):
        # Only intervene when G is a CPU-resident DTensor; otherwise the
        # original code is fine.
        if isinstance(G, _DTensor):
            local = G.to_local()
            if local.device.type == "cpu":
                cuda_dev = torch.device("cuda", torch.cuda.current_device())
                # IMPORTANT: move the DTensor itself to CUDA with `.to()`,
                # which preserves the full DTensorSpec — mesh, placements AND
                # the true global shape. Do NOT rebuild via `from_local`:
                # for an unevenly-sharded param (e.g. global dim0=3 over 4
                # ranks → torch.chunk gives [1,1,1] + an empty [0,*] on the
                # last rank), `from_local` re-infers the global shape as
                # local_dim0 * world_size, which differs across ranks
                # (4 vs 0) → the `full_tensor()` all-gather posts mismatched
                # sizes → NCCL hangs one rank forever.
                G_cuda = G.to(cuda_dev)
                # `_orig_ns5` does full_tensor (CUDA all-gather, correct under
                # uneven sharding because the spec is intact) → NS5 on CUDA
                # → wraps as Replicate DTensor with CUDA local.
                out_gpu = _orig_ns5(G_cuda, steps=steps)
                # Strip the DTensor wrapper; we apply the update directly on
                # the local in our patched step(). Return the *full* tensor
                # (true global shape) so step() can slice for its own shard.
                return out_gpu.full_tensor().to("cpu")
        return _orig_ns5(G, steps=steps)

    _muon_mod.zeropower_via_newtonschulz5 = _patched_ns5

    _orig_step = _muon_mod.Muon.step

    def _local_view(t):
        """Return the local tensor and a flag indicating whether the param
        is FSDP-cpu-offloaded (DTensor with CPU `_local_tensor`)."""
        if isinstance(t, _DTensor) and t._local_tensor.device.type == "cpu":
            return t._local_tensor, True
        return t, False

    def _slice_full_to_local(full_t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Take the full 2D Muon update `full_t` (Muon flattens g via
        `g.view(g.size(0), -1)` before NS5, so this is always 2D) and slice
        to this rank's local shard along dim 0, then reshape to
        `p._local_tensor.shape` (which may have higher rank, e.g. conv 5D).

        FSDP2 pads dim-0 of every shard to `ceil(N / world_size)`, so each
        rank owns exactly that many rows with zeros padded on the trailing
        rank(s) when N isn't divisible."""
        if not isinstance(p, _DTensor):
            return full_t.contiguous().view(p.shape)
        mesh = p.device_mesh
        # Find the shard dim (typically 0 for FSDP2) & local rank.
        local_rank = 0
        n_shards = 1
        for i, placement in enumerate(p.placements):
            if placement.is_shard():
                local_rank = mesh.get_local_rank(i)
                n_shards = mesh.size(i)
                break
        p_local_shape = p._local_tensor.shape
        if n_shards <= 1:
            return full_t.contiguous().view(p_local_shape)
        p_local_dim0 = p_local_shape[0]
        offset = local_rank * p_local_dim0
        end = min(offset + p_local_dim0, full_t.shape[0])
        shard_rows = full_t[offset:end] if offset < full_t.shape[0] else full_t[0:0]
        if shard_rows.shape[0] < p_local_dim0:
            pad = full_t.new_zeros((p_local_dim0,) + tuple(full_t.shape[1:]))
            pad[: shard_rows.shape[0]] = shard_rows
            shard_rows = pad
        return shard_rows.contiguous().view(p_local_shape)

    def _patched_step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # ----- Muon path -----
            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            for p in params:
                g = p.grad
                if g is None:
                    # Don't `continue` here: a Muon param whose grad is None
                    # (e.g. an input projection whose only forward path runs
                    # under no_grad — like the text-prefill projections) must
                    # still issue the same NS5 `full_tensor()` all-gather on
                    # every rank, otherwise the collective desyncs and the
                    # NCCL watchdog times out. Fall back to a zero grad.
                    #
                    # Use `torch.zeros_like(p)`, NOT `from_local(zeros_like(
                    # p._local_tensor), mesh, placements)`: for an unevenly
                    # sharded param the local shard sizes differ across ranks,
                    # so `from_local` re-infers a rank-divergent global shape
                    # → the later `full_tensor()` all-gather posts mismatched
                    # sizes → hang. `zeros_like` on a DTensor preserves p's
                    # exact DTensorSpec (mesh, placements, true global shape).
                    g = torch.zeros_like(p)
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                g = g.bfloat16()
                u = _muon_mod.zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                p_local, is_cpu_off = _local_view(p)
                if is_cpu_off:
                    # u from patched ns5 is a plain CPU 2D tensor (full,
                    # un-sharded). Slice to this rank's Shard(0) row range
                    # and view to match p._local_tensor's (possibly >2D) shape.
                    if isinstance(u, _DTensor):
                        u = u.full_tensor().to("cpu")
                    u_local = _slice_full_to_local(u, p)
                    p_local.mul_(1 - lr * wd)
                    p_local.add_(u_local, alpha=-adjusted_lr)
                else:
                    p.data.mul_(1 - lr * wd)
                    p.data.add_(u.view(p.shape), alpha=-adjusted_lr)

            # ----- AdamW backup path -----
            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step_n = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g_normed = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1 ** step_n
                bias_correction2 = 1 - beta2 ** step_n
                scale = bias_correction1 / bias_correction2 ** 0.5

                p_local, is_cpu_off = _local_view(p)
                if is_cpu_off:
                    g_local = (
                        g_normed._local_tensor
                        if isinstance(g_normed, _DTensor)
                        else g_normed
                    )
                    p_local.mul_(1 - lr * weight_decay)
                    p_local.add_(g_local, alpha=-lr / scale)
                else:
                    p.data.mul_(1 - lr * weight_decay)
                    p.data.add_(g_normed, alpha=-lr / scale)

        return loss

    _muon_mod.Muon.step = _patched_step
    _muon_mod._cpuoff_patched = True


_patch_muon_for_cpu_offload()


logger = init_logger(__name__)


def _build_fsdp_transformer(
    base_dir: str,
    action_ckpt: Optional[str],
    *,
    hsdp_replicate_dim: int,
    hsdp_shard_dim: int,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
    cpu_offload: bool = False,
) -> DMDHunyuanTransformer:
    """Load DMDHunyuanTransformer and wrap with FSDP `fully_shard` + HSDP mesh,
    mirroring `trainer/models/loader/fsdp_load.maybe_load_fsdp_model`:

      * fp32 master weights
      * bf16 forward / fp32 reduce via MixedPrecisionPolicy
      * hybrid_shard across (hsdp_replicate_dim, hsdp_shard_dim) device mesh
      * uses the transformer's `_fsdp_shard_conditions` (inherited from
        ARHunyuanVideo_1_5_DiffusionTransformer via DMDHunyuanTransformer)

    The inference-side `HunyuanVideo_1_5_DiffusionTransformer` exposes
    `add_action_parameters` (not `add_discrete_action_parameters`), so we
    inline the action+ckpt loading rather than going through the
    `cls_name`-string dispatch in `maybe_load_fsdp_model`.
    """
    from torch.distributed import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy
    from trainer.models.loader.fsdp_load import shard_model, set_default_dtype
    from trainer.utils import set_mixed_precision_policy

    mp_policy = MixedPrecisionPolicy(
        param_dtype, reduce_dtype, None, cast_forward_inputs=False
    )
    set_mixed_precision_policy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=None,
        mp_policy=mp_policy,
    )

    with set_default_dtype(param_dtype):
        logger.info("Loading DMDHunyuanTransformer base from: %s", base_dir)
        model = DMDHunyuanTransformer.from_pretrained(
            base_dir, local_attn_size=-1, sink_size=0
        )
        # The inference-side class names its action init `add_action_parameters`.
        model.add_action_parameters()
        if action_ckpt is not None and os.path.exists(action_ckpt):
            logger.info("Loading action ckpt: %s", action_ckpt)
            state_dict = load_file(action_ckpt)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if len(missing) > 0:
                logger.warning(
                    "Missing keys: %d (first 5: %s)", len(missing), missing[:5]
                )
            if len(unexpected) > 0:
                logger.warning(
                    "Unexpected keys: %d (first 5: %s)",
                    len(unexpected), unexpected[:5],
                )
        # Master weights live in fp32; MixedPrecisionPolicy casts to bf16 inside forward.
        model.to(torch.float32)

    device_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(hsdp_replicate_dim, hsdp_shard_dim),
        mesh_dim_names=("replicate", "shard"),
    )
    shard_model(
        model,
        cpu_offload=cpu_offload,
        reshard_after_forward=True,
        mp_policy=mp_policy,
        mesh=device_mesh,
        fsdp_shard_conditions=model._fsdp_shard_conditions,
        pin_cpu_memory=True,
    )
    return model


def _zero_uncond_tensor(t: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(t)


class DMDInitTrainingPipeline(TrainingPipeline):
    """Init-phase DMD training pipeline.

    Loads the scheduler via the base class (`_required_config_modules`) and
    builds generator / real / fake transformers manually in
    `initialize_training_pipeline()` to avoid having to register a new class
    with the trainer's model loader.
    """

    _required_config_modules = ["scheduler", "transformer"]

    def initialize_pipeline(self, trainer_args: TrainerArgs):
        pass

    def create_pipeline_stages(self, trainer_args: TrainerArgs):
        pass

    def create_training_stages(self, training_args: TrainingArgs):
        pass

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        pass

    # ------------------------------------------------------------------ setup

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing WorldPlay DMD init training pipeline...")
        self.device = get_local_torch_device()
        self.training_args = training_args
        world_group = get_world_group()
        self.world_size = world_group.world_size
        self.global_rank = world_group.rank
        self.sp_group = get_sp_group()
        self.rank_in_sp_group = self.sp_group.rank_in_group
        self.sp_world_size = self.sp_group.world_size
        self.local_rank = world_group.local_rank

        self.seed = training_args.seed
        set_random_seed(self.seed)
        self.set_schemas()
        self.action = training_args.action
        self.causal = training_args.causal
        self.train_time_shift = training_args.train_time_shift

        # Build the three transformers. They share architecture; we load
        # different state_dicts. All start on CPU then we cast to fp32 (Muon).
        base_dir = training_args.load_from_dir
        ar_ckpt = training_args.ar_action_load_from_dir
        # DMD-specific paths are passed via env vars (TrainingArgs dataclass doesn't
        # carry them through; FlexibleArgumentParser strips unknown fields).
        real_ckpt = (
            getattr(training_args, "real_score_load_from_dir", None)
            or os.environ.get("REAL_SCORE_LOAD_FROM_DIR")
        )
        fake_init_ckpt = (
            getattr(training_args, "fake_score_load_from_dir", None)
            or os.environ.get("FAKE_SCORE_LOAD_FROM_DIR")
            or real_ckpt
        )
        if real_ckpt is None:
            raise ValueError(
                "DMD init pipeline requires REAL_SCORE_LOAD_FROM_DIR env var "
                "(BI bidirectional checkpoint path)"
            )

        fsdp_kwargs = dict(
            hsdp_replicate_dim=training_args.hsdp_replicate_dim,
            hsdp_shard_dim=training_args.hsdp_shard_dim,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cpu_offload=bool(int(os.environ.get("DMD_FSDP_CPU_OFFLOAD", "1"))),
        )
        # Three transformers, each FSDP-wrapped with the same MixedPrecisionPolicy.
        # `_fsdp_shard_conditions` is inherited from the trainer-side AR class.
        self.generator = _build_fsdp_transformer(base_dir, ar_ckpt, **fsdp_kwargs)
        self.real_score = _build_fsdp_transformer(base_dir, real_ckpt, **fsdp_kwargs)
        self.real_score.eval()
        for p in self.real_score.parameters():
            p.requires_grad_(False)
        self.fake_score = _build_fsdp_transformer(base_dir, fake_init_ckpt, **fsdp_kwargs)

        if training_args.enable_gradient_checkpointing_type is not None:
            ckpt_t = training_args.enable_gradient_checkpointing_type
            self.generator = apply_activation_checkpointing(self.generator, checkpointing_type=ckpt_t)
            self.fake_score = apply_activation_checkpointing(self.fake_score, checkpointing_type=ckpt_t)

            # Verify activation checkpointing actually wrapped the inference-side
            # double_blocks (the inference transformer has both double_blocks and
            # single_blocks; HY's training-side AR transformer has only double_blocks).
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                CheckpointWrapper,
            )
            for name, model in (("generator", self.generator), ("fake_score", self.fake_score)):
                for attr in ("double_blocks", "single_blocks"):
                    blocks = getattr(model, attr, None)
                    if blocks is None:
                        continue
                    n_total = len(blocks)
                    n_wrap = sum(1 for b in blocks if isinstance(b, CheckpointWrapper))
                    if self.global_rank == 0:
                        logger.info("[ckpt verify] %s.%s wrapped=%d/%d", name, attr, n_wrap, n_total)

        # Scheduler: build a FlowMatchDiscreteScheduler whose `shift` matches
        # stage1 AR causal student training (run_ar_hunyuan_action_mem.sh uses
        # train_time_shift=3.0). The few-step denoising timestep grid the
        # self-forcing rollout runs on must match the shift the student was
        # pretrained at — otherwise the generator is distilled at timesteps
        # it was never trained on. Build the scheduler explicitly (not via
        # get_module("scheduler")) so `shift` is known and controllable.
        # NOTE: inference 480p_i2v default (commons/__init__.py) is 5.0; when
        # running this distill's output through generate.py you must override
        # with --flow_shift 3.0.
        from hyvideo.schedulers.scheduling_flow_match_discrete import (
            FlowMatchDiscreteScheduler,
        )
        # `train_time_shift` is the knob; default 3.0 == stage1 shift.
        flow_shift = getattr(training_args, "train_time_shift", 3.0)
        self.noise_scheduler = FlowMatchDiscreteScheduler(
            shift=flow_shift, reverse=True, solver="euler"
        )
        # Derive the 4-step denoising timestep list from the shifted scheduler so
        # it is identical to what worldplay_video_pipeline._ar_rollout_inner
        # iterates when run with the same shift.
        num_inference_steps = 4
        self.noise_scheduler.set_timesteps(num_inference_steps)
        denoising_step_list = [float(t) for t in self.noise_scheduler.timesteps.tolist()]
        if self.global_rank == 0:
            logger.info(
                "DMD denoising schedule: flow_shift=%.2f -> %s",
                flow_shift, denoising_step_list,
            )

        # Self-forcing rollout + DMD wrapper
        # SLICE_LAST_FRAMES env: integer for "only last N latents have grad",
        # unset/empty → all chunks have grad (LongLive train_init).
        slice_last_env = os.environ.get("SLICE_LAST_FRAMES", "").strip()
        slice_last = int(slice_last_env) if slice_last_env else None
        if slice_last is not None and self.global_rank == 0:
            logger.info("Self-forcing slice_last_frames=%d (only last chunk(s) grad)", slice_last)
        memory_frames_env = int(os.environ.get("MEMORY_FRAMES", "20"))
        # Stage1-parity memory-noise injection (CameraJsonWMemDataset
        # select_window_out_flag==1 reproduction). Default 0.8 matches the
        # 80% rate at which stage1 trained the AR student under heavily-noised
        # memory KV. Set MEMORY_NOISE_PROB=0 to disable.
        memory_noise_prob_env = float(os.environ.get("MEMORY_NOISE_PROB", "0.8"))
        # These env names are kept for compatibility, but the values are stage1
        # scheduler index bounds, matching `torch.randint(500, 985)` in
        # ar_hunyuan_mem_training_pipeline.py. They are converted to true model
        # timesteps inside WorldPlaySelfForcingPipeline.
        memory_noise_t_min_env = int(os.environ.get("MEMORY_NOISE_T_MIN", "500"))
        memory_noise_t_max_env = int(os.environ.get("MEMORY_NOISE_T_MAX", "985"))
        if self.global_rank == 0:
            logger.info(
                "Self-forcing memory config: memory_frames=%d "
                "memory_noise_prob=%.2f stage1_index_range=[%d, %d)",
                memory_frames_env,
                memory_noise_prob_env,
                memory_noise_t_min_env,
                memory_noise_t_max_env,
            )
        self.self_forcing_pipe = WorldPlaySelfForcingPipeline(
            generator=self.generator,
            scheduler=self.noise_scheduler,
            denoising_step_list=denoising_step_list,
            num_train_timestep=1000,
            chunk_latent_frames=4,
            num_inference_steps=num_inference_steps,
            slice_last_frames=slice_last,
            memory_frames=memory_frames_env,
            memory_noise_prob=memory_noise_prob_env,
            memory_noise_t_min=memory_noise_t_min_env,
            memory_noise_t_max=memory_noise_t_max_env,
        )
        real_guidance_scale = float(
            os.environ.get(
                "REAL_GUIDANCE_SCALE",
                getattr(training_args, "real_score_guidance_scale", 6.0),
            )
        )
        fake_guidance_scale = float(
            os.environ.get(
                "FAKE_GUIDANCE_SCALE",
                getattr(training_args, "fake_guidance_scale", 0.0),
            )
        )
        min_score_ratio = float(
            os.environ.get(
                "MIN_TIMESTEP_RATIO",
                getattr(training_args, "min_timestep_ratio", 0.2),
            )
        )
        max_score_ratio = float(
            os.environ.get(
                "MAX_TIMESTEP_RATIO",
                getattr(training_args, "max_timestep_ratio", 0.98),
            )
        )
        if self.global_rank == 0:
            logger.info(
                "DMD score config: min_timestep_ratio=%.3f max_timestep_ratio=%.3f "
                "real_guidance_scale=%.3f fake_guidance_scale=%.3f",
                min_score_ratio,
                max_score_ratio,
                real_guidance_scale,
                fake_guidance_scale,
            )

        self.dmd = WorldPlayDMD(
            generator=self.generator,
            real_score=self.real_score,
            fake_score=self.fake_score,
            scheduler=self.noise_scheduler,
            self_forcing_pipeline=self.self_forcing_pipe,
            num_train_timestep=1000,
            denoising_step_list=denoising_step_list,
            real_guidance_scale=real_guidance_scale,
            fake_guidance_scale=fake_guidance_scale,
            min_score_ratio=min_score_ratio,
            max_score_ratio=max_score_ratio,
            # DMD re-noise timestep (see WorldPlayDMD._re_noise): ts_schedule
            # off -> uniform-sample [0, 1000], apply the SD3 shift (= flow_shift),
            # clamp to [min_timestep_ratio, max_timestep_ratio] — matches LongLive train_init (ts_schedule=
            # false, timestep_shift=5.0). The old ts_schedule=True path tied the
            # band to the rollout exit-step indices which — with a descending
            # denoising_step_list — collapsed `_re_noise` to t≈999 every step.
            ts_schedule=False,
            timestep_shift=flow_shift,
            chunk_latent_frames=4,
            device=self.device,
        )

        # Optimizers: Muon for both trainable transformers, matching
        # ar_hunyuan_mem_training_pipeline.py.
        self.gen_optimizer = get_muon_optimizer(
            model=self.generator,
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
            adamw_betas=(0.9, 0.999),
            adamw_eps=1e-8,
        )
        critic_lr = (
            getattr(training_args, "critic_learning_rate", None)
            or float(os.environ.get("CRITIC_LR", training_args.learning_rate * 0.5))
        )
        self.crit_optimizer = get_muon_optimizer(
            model=self.fake_score,
            lr=critic_lr,
            weight_decay=training_args.weight_decay,
            adamw_betas=(0.9, 0.999),
            adamw_eps=1e-8,
        )

        # Reuse base class for LR scheduler on the generator
        from trainer.training.training_utils import get_scheduler
        self.lr_scheduler = get_scheduler(
            training_args.lr_scheduler,
            optimizer=self.gen_optimizer,
            num_warmup_steps=training_args.lr_warmup_steps,
            num_training_steps=training_args.max_train_steps,
            num_cycles=training_args.lr_num_cycles,
            power=training_args.lr_power,
            min_lr_ratio=training_args.min_lr_ratio,
            last_epoch=-1,
        )

        # The inference-side transformer reads a module-global InferState. It's
        # normally initialized by `hyvideo.commons.infer_state.initialize_infer_state`
        # from the CLI args of `hyvideo/generate.py`. In training we don't go
        # through that entrypoint, so set a sensible default explicitly.
        import hyvideo.commons.infer_state as _infer_state_mod
        setattr(_infer_state_mod, "_InferState__infer_state", _infer_state_mod.InferState())
        # the module-level name in infer_state.py is literally `__infer_state`;
        # python module-level dunder names are NOT mangled (only class attrs are),
        # so we need to set the unmangled attribute via vars().
        vars(_infer_state_mod)["__infer_state"] = _infer_state_mod.InferState()
        # Fallback: monkey-patch get_infer_state directly.
        _default_state = _infer_state_mod.InferState()
        _infer_state_mod.get_infer_state = lambda: _default_state

        self.num_training_frames = int(
            os.environ.get("NUM_TRAINING_FRAMES")
            or getattr(training_args, "num_training_frames", None)
            or 32
        )

        # Dataloader — DMD-specific reader that forces mode-B (chronological
        # first-N frames, no memory-frame reshuffling).
        from trainer.training.dmd.dmd_dataset import build_dmd_dataloader
        self.train_dataset, self.train_dataloader = build_dmd_dataloader(
            json_path=training_args.json_path,
            num_training_frames=self.num_training_frames,
            batch_size=training_args.train_batch_size,
            num_workers=training_args.dataloader_num_workers,
            cfg_rate=training_args.training_cfg_rate,
            seed=self.seed,
            drop_last=True,
        )
        self.train_loader_iter = iter(self.train_dataloader)

        self.dfake_gen_update_ratio = int(
            os.environ.get("DFAKE_GEN_UPDATE_RATIO")
            or getattr(training_args, "dfake_gen_update_ratio", None)
            or 5
        )
        if self.global_rank == 0:
            logger.info(
                "DMD rollout config: num_training_frames=%d dfake_gen_update_ratio=%d",
                self.num_training_frames,
                self.dfake_gen_update_ratio,
            )
        # Validation: every VALIDATION_EVERY steps run a deterministic 4-step
        # inference rollout on one fixed training sample and save the latent.
        self.validate_every = int(os.environ.get("VALIDATION_EVERY", "100"))
        self.val_sample_idx = int(os.environ.get("VALIDATION_SAMPLE_IDX", "0"))
        self.init_steps = 0
        # Restore weights + step counter if RESUME_FROM / resume_from_checkpoint
        # points at a DCP checkpoint dir. Collective — all ranks participate.
        self._maybe_resume_dmd()

        if self.global_rank == 0:
            project = training_args.tracker_project_name or "worldplay-dmd-init"
            wandb_config = dataclasses.asdict(training_args)
            if training_args.wandb_key:
                wandb.login(key=training_args.wandb_key)
            wandb.init(
                config=wandb_config,
                name=training_args.wandb_run_name,
                entity=training_args.wandb_entity,
                project=project,
            )

    # ----------------------------------------------------------- batch + step

    def _next_dmd_batch(self) -> Dict[str, torch.Tensor]:
        try:
            raw = next(self.train_loader_iter)
        except StopIteration:
            self.train_loader_iter = iter(self.train_dataloader)
            raw = next(self.train_loader_iter)
        return self._assemble_dmd_batch(raw)

    def _assemble_dmd_batch(
        self, raw: Dict, initial_noise: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Move a collated raw batch to device/dtype, sample the initial noise
        canvas (unless `initial_noise` is supplied — validation passes a fixed
        one), and build the unconditional twin."""
        dev = self.device
        dtype = torch.bfloat16

        # `dmd_dataset.DMDDataset` already returns:
        #   cond_latents [B, C+1, T, H, W]   (image_cond at frame 0 + i2v mask channel)
        #   w2c, intrinsic, action            chronologically first T frames
        cond_latents = raw["cond_latents"].to(dev, dtype=dtype)
        # viewmats / Ks stay float32 — bf16 rounds camera matrices badly (the
        # camera-center translation deltas are only a few units, and bf16 keeps
        # ~2-3 significant figures), which corrupts `select_aligned_memory_frames`.
        # Real inference (`generate.py` / `_ar_rollout_inner`) carries float32
        # camera and bf16-casts only at the transformer forward — the rollout's
        # `.to(target_dtype)` does the same, so training must feed float32 here.
        viewmats = raw["w2c"].to(dev, dtype=torch.float32)
        Ks = raw["intrinsic"].to(dev, dtype=torch.float32)
        action = raw["action"].to(dev, dtype=dtype)
        prompt_embeds = raw["prompt_embed"].to(dev, dtype=dtype)
        prompt_mask = raw["prompt_mask"].to(dev, dtype=dtype)
        vision_states = raw["vision_states"].to(dev, dtype=dtype)
        byt5_text_states = raw["byt5_text_states"].to(dev, dtype=dtype)
        byt5_text_mask = raw["byt5_text_mask"].to(dev, dtype=dtype)

        B, C_plus_1, T, H, W = cond_latents.shape
        C = C_plus_1 - 1     # strip the i2v mask channel for the noise canvas
        if initial_noise is None:
            initial_noise = torch.randn(B, C, T, H, W, device=dev, dtype=dtype)
        else:
            initial_noise = initial_noise.to(dev, dtype=dtype)

        batch = {
            "initial_noise": initial_noise,
            "cond_latents": cond_latents,
            "viewmats": viewmats,
            "Ks": Ks,
            "action": action,
            "prompt_embeds": prompt_embeds,
            "prompt_mask": prompt_mask,
            "vision_states": vision_states,
            "byt5_text_states": byt5_text_states,
            "byt5_text_mask": byt5_text_mask,
        }
        # Build unconditional twin (zeros for text/byt5/vision conditioning,
        # keeping action/camera since those still apply to neg branch).
        for k_pos, k_neg in [
            ("prompt_embeds", "uncond_prompt_embeds"),
            ("prompt_mask", "uncond_prompt_mask"),
            ("vision_states", "uncond_vision_states"),
            ("byt5_text_states", "uncond_byt5_text_states"),
            ("byt5_text_mask", "uncond_byt5_text_mask"),
        ]:
            batch[k_neg] = _zero_uncond_tensor(batch[k_pos])

        return batch

    def _make_val_batch(self) -> Dict[str, torch.Tensor]:
        """Build the batch for the fixed validation sample.

        Action label already matches inference: `ar_camera_hunyuan_w_mem_dataset`
        now derives `raw["action"]` from raw pose.json via the same
        `pose_to_input()` that `generate.py` uses, so no override needed here.

        Initial-noise seed: env `VALIDATION_SEED` (default 1, matching
        `run_ar_50step.sh --seed 1`). Same generator construction as
        `worldplay_video_pipeline:1661`. Lets validation be byte-compared to
        `generate.py --num_inference_steps 4 --seed 1` with live student weights.
        """
        from torch.utils.data import default_collate

        raw = default_collate([self.train_dataset[self.val_sample_idx]])
        _, C_plus_1, T, H, W = raw["cond_latents"].shape

        val_seed = int(os.environ.get("VALIDATION_SEED", "1"))
        g = torch.Generator(device=self.device).manual_seed(val_seed)
        noise = torch.randn(
            1, C_plus_1 - 1, T, H, W,
            device=self.device, dtype=torch.bfloat16, generator=g,
        )
        return self._assemble_dmd_batch(raw, initial_noise=noise)

    def _get_val_vae(self):
        """Lazy-load the HunyuanVideo-1.5 VAE (rank 0 only) for decoding the
        validation rollout to video. Loaded once, then cached."""
        if getattr(self, "_val_vae", None) is not None:
            return self._val_vae
        from hyvideo.models.autoencoders.hunyuanvideo_15_vae_w_cache import (
            AutoencoderKLConv3D,
        )
        vae_dir = os.path.join(self.training_args.model_path, "vae")
        logger.info("Loading VAE for validation decode from %s", vae_dir)
        vae = AutoencoderKLConv3D.from_pretrained(vae_dir, torch_dtype=torch.float32)
        vae = vae.to(self.device).eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        self._val_vae = vae
        return vae

    @torch.no_grad()
    def _decode_latent_to_mp4(self, latent: torch.Tensor, out_path: str) -> None:
        """Decode a [1, C, T, H, W] latent to an mp4. Mirrors
        worldplay_video_pipeline: latent / scaling_factor -> vae.decode ->
        [-1,1] -> [0,1] -> uint8 THWC -> imageio mimwrite @ 24fps."""
        import einops
        import imageio

        vae = self._get_val_vae()
        z = latent.to(self.device, dtype=torch.float32)
        if getattr(vae.config, "shift_factor", None):
            z = z / vae.config.scaling_factor + vae.config.shift_factor
        else:
            z = z / vae.config.scaling_factor
        frames = vae.decode(z, return_dict=False)[0]          # [B,3,T,H,W] in [-1,1]
        frames = (frames / 2 + 0.5).clamp(0, 1).float().cpu()
        if frames.ndim == 5:
            frames = frames[0]                                # [3,T,H,W]
        vid = (frames * 255).clamp(0, 255).to(torch.uint8)
        vid = einops.rearrange(vid, "c f h w -> f h w c").numpy()
        imageio.mimwrite(out_path, vid, fps=16)  # GameFactory data is 16 fps

    def _build_ar_rollout_shim(self, latent_frames: int):
        """Minimal stand-in for `HunyuanVideo_1_5_Pipeline` so validation runs
        the REAL, unmodified inference rollout `_ar_rollout_inner` against the
        live `self.generator` weights.

        Why a shim and not `HunyuanVideo_1_5_Pipeline.create_pipeline(...)`:
        create_pipeline eagerly loads a *second* full transformer (+ VAE, text /
        byt5 / vision encoders). Training already sits at ~105GB of 140GB per
        GPU — a second transformer OOMs. The shim is a real
        `HunyuanVideo_1_5_Pipeline` built via `object.__new__` (skipping
        `__init__`, which loads models); because it IS the class,
        `_ar_rollout_inner`, `init_kv_cache`, `progress_bar` and the
        `guidance_scale` / `do_classifier_free_guidance` properties all resolve
        to the genuine inference methods — the rollout is byte-for-byte the
        `generate.py --model_type ar --few_step true --num_inference_steps 4`
        code path. We set exactly the `self.<attr>` those methods touch
        (verified against `_ar_rollout_inner` + `init_kv_cache`)."""
        from hyvideo.pipelines.worldplay_video_pipeline import (
            HunyuanVideo_1_5_Pipeline,
        )
        from hyvideo.pipelines.pipeline_utils import retrieve_timesteps
        from hyvideo.schedulers.scheduling_flow_match_discrete import (
            FlowMatchDiscreteScheduler,
        )
        from hyvideo.utils.retrieval_context import generate_points_in_sphere

        # diffusers' DiffusionPipeline.__setattr__ consults `self.config` on
        # every attribute *re-assignment* — and `_ar_rollout_inner` re-assigns
        # `self._kv_cache` once per chunk. A bare `object.__new__` shim has no
        # `.config`, so subclass with a plain __setattr__ to bypass that
        # machinery entirely. None of `_ar_rollout_inner` / `init_kv_cache` /
        # `progress_bar` ever reads `self.config`.
        class _ARRolloutShim(HunyuanVideo_1_5_Pipeline):
            def __setattr__(self, name, value):
                object.__setattr__(self, name, value)

        shim = object.__new__(_ARRolloutShim)
        shim.transformer = self.generator              # live, being-trained weights
        shim.execution_device = torch.device(self.device)
        shim.enable_offloading = False                 # auto_offload_model -> no-op
        shim.target_dtype = torch.bfloat16
        shim.autocast_enabled = True
        shim._progress_bar_config = {}
        shim._guidance_scale = 1.0                     # few_step => do_cfg is False

        num_inference_steps = 4
        scheduler = FlowMatchDiscreteScheduler(shift=5.0, reverse=True, solver="euler")
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler, num_inference_steps, self.device
        )
        shim.scheduler = scheduler
        shim.num_inference_steps = num_inference_steps
        shim.num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order
        shim.chunk_latent_frames = 4
        shim.chunk_num = latent_frames // 4            # `_ar_rollout_inner` loops range(chunk_num)
        shim.points_local = generate_points_in_sphere(50000, 8.0).to(self.device)
        return shim, timesteps

    @torch.no_grad()
    def _validate(self, step: int) -> None:
        """Every `validate_every` steps: generate one fixed-sample video with the
        current generator via the EXACT inference rollout
        (`HunyuanVideo_1_5_Pipeline._ar_rollout_inner` — the `run_ar_4step.sh`
        code path), VAE-decode it to an mp4, and save the raw latent alongside.

        Collective: the FSDP generator forwards inside `_ar_rollout_inner`
        all-gather across every rank, so all ranks run the rollout identically
        (the val batch + the seeded `points_local` are identical on each rank);
        only rank 0 decodes + writes. The VAE decode is wrapped in try/except so
        a decode failure can never take down training."""
        from hyvideo.pipelines.worldplay_video_pipeline import (
            HunyuanVideo_1_5_Pipeline,
        )
        # The whole body is wrapped: a validation failure must never take down
        # training — it just logs a warning and is skipped. Safe even though the
        # rollout is collective: all ranks run it on identical inputs (same val
        # batch, same seeded `points_local`), so any exception is deterministic
        # — every rank raises and catches together — and the final
        # `dist.barrier()` (outside the try) keeps them in lockstep.
        try:
            batch = self._make_val_batch()
            latents = batch["initial_noise"].clone()  # _ar_rollout_inner mutates in place
            extra_kwargs = {
                "byt5_text_states": batch["byt5_text_states"],
                "byt5_text_mask": batch["byt5_text_mask"],
            }
            # Seed + fork RNG so the rollout is deterministic across validations
            # (only the weights change) and identical on every rank.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(int(os.environ.get("VALIDATION_SEED", "1")))
                shim, timesteps = self._build_ar_rollout_shim(latents.shape[2])
                clean = HunyuanVideo_1_5_Pipeline._ar_rollout_inner(
                    shim,
                    latents=latents,
                    timesteps=timesteps,
                    prompt_embeds=batch["prompt_embeds"],
                    prompt_mask=batch["prompt_mask"],
                    vision_states=batch["vision_states"],
                    cond_latents=batch["cond_latents"],
                    task_type="i2v",
                    extra_kwargs=extra_kwargs,
                    # viewmats / Ks go in as float32, NOT the batch's bf16:
                    # `_ar_rollout_inner` does `viewmats[0].cpu().numpy()` for the
                    # geometry-based memory-frame selection and numpy has no
                    # bfloat16. Real inference also passes float camera matrices;
                    # they are cast to the model dtype internally where needed.
                    viewmats=batch["viewmats"].float(),
                    Ks=batch["Ks"].float(),
                    action=batch["action"],
                    device=self.device,
                )
            abs_mean = float(clean.detach().abs().mean())
            if self.global_rank == 0:
                val_dir = os.path.join(self.training_args.output_dir, "val")
                os.makedirs(val_dir, exist_ok=True)
                torch.save(
                    {
                        "step": step,
                        "val_sample_idx": self.val_sample_idx,
                        "latent": clean.detach().float().cpu(),
                        "abs_mean": abs_mean,
                    },
                    os.path.join(val_dir, f"step_{step:06d}_latent.pt"),
                )
                mp4_path = os.path.join(val_dir, f"step_{step:06d}.mp4")
                self._decode_latent_to_mp4(clean.detach(), mp4_path)
                logger.info(
                    "Validation @ step %d: latent abs_mean=%.4f, video -> %s",
                    step, abs_mean, mp4_path,
                )
        except Exception as e:  # validation must never kill training
            if self.global_rank == 0:
                import traceback as _tb
                logger.warning(
                    "Validation @ step %d FAILED — training continues. %r\n%s",
                    step, e, _tb.format_exc(),
                )
        if dist.is_initialized():
            dist.barrier()

    def train_one_dmd_step(self, step: int) -> Dict[str, Any]:
        """One alternating-update step (LongLive `Trainer.train` distillation
        path, non-streaming branch). Returns merged log dict."""
        do_gen = (step % self.dfake_gen_update_ratio == 0)
        log: Dict[str, Any] = {"step": step, "do_gen": int(do_gen)}

        batch = self._next_dmd_batch()

        if do_gen:
            self.gen_optimizer.zero_grad(set_to_none=True)
            gen_loss, gen_log = self.dmd.generator_loss(batch, target_dtype=torch.bfloat16)
            gen_loss.backward()
            gen_grad = clip_grad_norm_while_handling_failing_dtensor_cases(
                [p for p in self.generator.parameters() if p.requires_grad],
                self.training_args.max_grad_norm,
            )
            self.gen_optimizer.step()
            self.lr_scheduler.step()
            log["gen_loss"] = float(gen_loss.detach())
            log["gen_grad_norm"] = float(gen_grad)
            log.update({f"gen/{k}": float(v) if torch.is_tensor(v) else v for k, v in gen_log.items()})

        # Critic step every iteration
        self.crit_optimizer.zero_grad(set_to_none=True)
        crit_loss, crit_log = self.dmd.critic_loss(batch, target_dtype=torch.bfloat16)
        crit_loss.backward()
        crit_grad = clip_grad_norm_while_handling_failing_dtensor_cases(
            [p for p in self.fake_score.parameters() if p.requires_grad],
            self.training_args.max_grad_norm,
        )
        self.crit_optimizer.step()
        log["crit_loss"] = float(crit_loss.detach())
        log["crit_grad_norm"] = float(crit_grad)
        log.update({f"crit/{k}": float(v) if torch.is_tensor(v) else v for k, v in crit_log.items()})

        return log

    # ---------------------------------------------------------------- driver

    def train(self) -> None:
        max_steps = self.training_args.max_train_steps
        log_iters = max(1, self.training_args.checkpointing_steps // 10)
        # Validate the starting weights once before training — gives a baseline
        # video for the freshly-loaded weights (step 0 on a fresh run, or the
        # resumed step, e.g. 100, on a checkpoint resume).
        if self.validate_every > 0:
            self._validate(self.init_steps)
        for step in range(self.init_steps, max_steps):
            log = self.train_one_dmd_step(step)
            if self.global_rank == 0 and (step % log_iters == 0):
                logger.info("step=%s log=%s", step, log)
                if wandb.run is not None:
                    wandb.log(log, step=step)
            if (step + 1) % self.training_args.checkpointing_steps == 0:
                self._save_dmd_checkpoint(step + 1)
            if self.validate_every > 0 and (step + 1) % self.validate_every == 0:
                self._validate(step + 1)
        self._save_dmd_checkpoint(max_steps)
        logger.info("DMD init training done.")

    def _save_dmd_checkpoint(self, step: int) -> None:
        """Save a COMPLETE, resumable checkpoint via torch.distributed.checkpoint.

        MUST run on every rank: `dcp.save` + `get_model_state_dict` are
        collective — they coordinate the FSDP shards across ranks. The old
        rank-0-only `state_dict()` + `torch.save` desynced the FSDP process
        group (rank 0 issued an all-gather the others didn't) → the next
        collective deadlocked → NCCL watchdog killed the run at step 100; it
        also only persisted rank 0's 1/4 shard (unrecoverable).

        DCP writes the model sharded across ranks into `out_dir`; loadable
        back with `dcp.load`. Optimizer (Muon momentum) state is intentionally
        not saved — disk budget — so resume restores weights only.
        """
        import json
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_model_state_dict

        out_dir = os.path.join(self.training_args.output_dir, f"checkpoint-{step}")
        # Prune to keep (checkpoints_total_limit - 1) old checkpoints BEFORE
        # writing the new one — so after save, total = checkpoints_total_limit.
        # Honors --checkpoints_total_limit (e.g. 3 → keep 2 old + 1 new = 3 total).
        # Requires peak disk = (total_limit) * ~64GB transient during save.
        # Pod NV is 244GB free; (3 * 64GB) = 192GB peak, safe with 52GB slack.
        # (Old behavior keep=0 deleted ALL before save — broke total_limit.)
        if self.global_rank == 0:
            keep_n = max(0, int(getattr(self.training_args, "checkpoints_total_limit", 1)) - 1)
            self._prune_old_checkpoints(keep=keep_n)
        if dist.is_initialized():
            dist.barrier()
        state = {
            "generator": get_model_state_dict(self.generator),
            "fake_score": get_model_state_dict(self.fake_score),
        }
        dcp.save(state, checkpoint_id=out_dir)  # collective: sharded write
        if self.global_rank == 0:
            with open(os.path.join(out_dir, "meta.json"), "w") as f:
                json.dump({"step": step}, f)
            logger.info("Saved DMD checkpoint to %s (step=%d)", out_dir, step)
        if dist.is_initialized():
            dist.barrier()

    def _prune_old_checkpoints(self, keep: int = 1) -> None:
        """Keep only the newest `keep` checkpoint-* dirs (/raid is near full)."""
        import re as _re
        import shutil
        base = self.training_args.output_dir
        cks = []
        for name in os.listdir(base):
            m = _re.fullmatch(r"checkpoint-(\d+)", name)
            if m and os.path.isdir(os.path.join(base, name)):
                cks.append((int(m.group(1)), name))
        cks.sort()
        # keep>0: drop all but the newest `keep`. keep==0: drop ALL (used by
        # `_save_dmd_checkpoint` to clear room before writing the new one).
        # Note `cks[:-0]` is empty, not the full list — hence the explicit branch.
        for _, name in (cks[:-keep] if keep > 0 else cks):
            shutil.rmtree(os.path.join(base, name), ignore_errors=True)
            logger.info("Pruned old checkpoint %s", name)

    def _maybe_resume_dmd(self) -> None:
        """Restore generator + fake_score weights from a DCP checkpoint dir
        (env `RESUME_FROM` or `--resume_from_checkpoint`) and set `init_steps`.

        Collective — every rank participates in `dcp.load`. Optimizer state is
        not restored (not saved); Muon momentum cold-starts on resume, a minor
        one-time perturbation acceptable for crash recovery."""
        import json
        resume = (
            os.environ.get("RESUME_FROM", "").strip()
            or getattr(self.training_args, "resume_from_checkpoint", None)
        )
        if not resume:
            return
        if not os.path.isdir(resume):
            logger.warning("RESUME_FROM=%s is not a directory; skip resume", resume)
            return
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            set_model_state_dict,
        )
        state = {
            "generator": get_model_state_dict(self.generator),
            "fake_score": get_model_state_dict(self.fake_score),
        }
        dcp.load(state, checkpoint_id=resume)  # collective: in-place load
        set_model_state_dict(self.generator, state["generator"])
        set_model_state_dict(self.fake_score, state["fake_score"])
        meta_path = os.path.join(resume, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.init_steps = int(json.load(f).get("step", 0))
        logger.info("Resumed from %s -> init_steps=%d", resume, self.init_steps)


def main(args) -> None:
    logger.info("Starting WorldPlay DMD init training pipeline...")
    pipeline = DMDInitTrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args
    )
    pipeline.train()
    logger.info("DMD init pipeline done")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = TrainerArgs.add_cli_args(parser)

    # DMD-specific extra args (registered via setattr after parse since
    # TrainingArgs is dataclass-frozen elsewhere).
    parser.add_argument("--real-score-load-from-dir", dest="real_score_load_from_dir", type=str, default=None,
                        help="BI bidirectional teacher safetensors path")
    parser.add_argument("--fake-score-load-from-dir", dest="fake_score_load_from_dir", type=str, default=None,
                        help="Optional explicit init for critic (defaults to real_score path)")
    parser.add_argument("--num-training-frames", dest="num_training_frames", type=int, default=32,
                        help="Latent frames in one self-forcing rollout (32 = ~5s)")
    parser.add_argument("--dfake-gen-update-ratio", dest="dfake_gen_update_ratio", type=int, default=5,
                        help="Generator step every N critic steps (LongLive default 5)")
    parser.add_argument("--real-guidance-scale", dest="real_guidance_scale", type=float, default=6.0)
    parser.add_argument("--fake-guidance-scale", dest="fake_guidance_scale", type=float, default=0.0)
    parser.add_argument("--critic-learning-rate", dest="critic_learning_rate", type=float, default=None)

    args = parser.parse_args()
    args.dit_cpu_offload = False
    main(args)
