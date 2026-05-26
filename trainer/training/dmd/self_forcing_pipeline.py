# SPDX-License-Identifier: Apache-2.0
"""
Self-forcing rollout pipeline for DMD distillation.

Algorithm follows LongLive's `SelfForcingTrainingPipeline.inference_with_trajectory`
adapted to WorldPlay's chunk-by-chunk causal generator + WorldPlay's existing
per-chunk memory-frame prefill KV-cache mechanism (which is left unchanged).

For each chunk:
  (1) optionally prefill kv_cache by forwarding `select_aligned_memory_frames`
      with cache_vision=True (no_grad).
  (2) run num_inference_steps denoising steps. A single step is randomly chosen
      as the "exit step": its model forward is gradient-enabled and its x0_pred
      is taken as the chunk's final clean latent. All other steps are no_grad.

The text/byt5/siglip cache is filled once at the start of the rollout via
forward_txt(cache_txt=True).

The pipeline returns (clean_latents, gradient_mask, denoised_timestep_from,
denoised_timestep_to) so the DMD loss can re-noise and score the rollout output.
"""

from typing import Optional, Tuple, List, Dict, Any
import os
import torch
import torch.distributed as dist

from hyvideo.utils.retrieval_context import (
    generate_points_in_sphere,
    select_aligned_memory_frames,
)


def _broadcast_bool(value: bool, device) -> bool:
    """Sample once on rank 0, broadcast to all ranks. Returns a Python bool."""
    if not dist.is_initialized():
        return bool(value)
    rank = dist.get_rank()
    if rank == 0:
        t = torch.tensor([1 if value else 0], device=device, dtype=torch.long)
    else:
        t = torch.empty(1, device=device, dtype=torch.long)
    dist.broadcast(t, src=0)
    return bool(t.item())


def _broadcast_long_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """In-place broadcast of an integer tensor from rank 0. Returns the tensor."""
    if dist.is_initialized():
        dist.broadcast(tensor, src=0)
    return tensor


def _zero_kv_cache_list(num_blocks: int) -> List[Dict[str, Optional[torch.Tensor]]]:
    return [
        {"k_vision": None, "v_vision": None, "k_txt": None, "v_txt": None}
        for _ in range(num_blocks)
    ]


def _broadcast_exit_step(num_chunks: int, num_steps: int, device, generator=None) -> List[int]:
    """Sample one exit step per chunk, synchronized across ranks."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        idx = torch.randint(0, num_steps, (num_chunks,), device=device, generator=generator)
    else:
        idx = torch.empty(num_chunks, dtype=torch.long, device=device)
    if dist.is_initialized():
        dist.broadcast(idx, src=0)
    return idx.tolist()


class WorldPlaySelfForcingPipeline:
    """Self-forcing rollout for a single training step.

    Holds references to the generator transformer + scheduler. KV cache is
    *not* persisted across calls — every `rollout()` re-initializes it so each
    training step starts fresh (this matches LongLive train_init semantics; the
    streaming/long variant lives in a separate pipeline).
    """

    def __init__(
        self,
        generator,                # DMDHunyuanTransformer (FSDP-wrapped is OK)
        scheduler,                # FlowMatchEulerDiscreteScheduler
        denoising_step_list,      # descending; derived from the shift-matched scheduler
        num_train_timestep: int = 1000,
        chunk_latent_frames: int = 4,
        num_inference_steps: int = 4,
        # HY memory-frame prefill knobs (defaults match worldplay_video_pipeline)
        memory_frames: int = 20,
        temporal_context_size: int = 12,
        pred_latent_size: int = 4,
        stabilization_level: int = 15,
        # Self-forcing grad control. `slice_last_frames=None` (default) → all
        # chunks have grad (matches LongLive train_init). Set to a positive int
        # to enable grad only on the last `slice_last_frames` latents, in
        # multiples of chunk_latent_frames (LongLive train_long pattern). Useful
        # for staying under memory at long T.
        slice_last_frames: int | None = None,
        # Stage1-parity memory-noise injection. CameraJsonWMemDataset's
        # select_window_out_flag==1 branch (~80% of training samples) overrides
        # non-last-chunk *scheduler indices* with randint(500, 985). In the
        # default 1000-step FlowMatchEuler schedule, index 500 -> timestep 500
        # and index 984 -> a low timestep. Reproduce that here by, with prob
        # `memory_noise_prob`, re-noising _prefill_memory's context latents at
        # per-source-chunk sampled scheduler indices in [idx_min, idx_max).
        # Disabled (prob=0.0) → original behavior (clean x0_pred + fixed
        # stabilization_level-1 timestep, matches inference _ar_rollout_inner).
        memory_noise_prob: float = 0.0,
        memory_noise_t_min: int = 500,
        memory_noise_t_max: int = 985,
    ):
        self.generator = generator
        self.scheduler = scheduler
        self.denoising_step_list = list(denoising_step_list)
        self.num_train_timestep = num_train_timestep
        self.chunk_latent_frames = chunk_latent_frames
        self.num_inference_steps = num_inference_steps
        self.memory_frames = memory_frames
        self.temporal_context_size = temporal_context_size
        self.pred_latent_size = pred_latent_size
        self.stabilization_level = stabilization_level
        self.slice_last_frames = slice_last_frames
        self.memory_noise_prob = float(memory_noise_prob)
        # Names kept for launcher compatibility, but these values are scheduler
        # index bounds, not literal model timesteps.
        self.memory_noise_idx_min = int(memory_noise_t_min)
        self.memory_noise_idx_max = int(memory_noise_t_max)

        # `select_aligned_memory_frames` needs a pre-computed point cloud anchor.
        # Generate once and stash.
        self._points_local = None

    @torch.no_grad()
    def _ensure_points_local(self, device):
        if self._points_local is None:
            # Same args as CameraJsonWMemDataset: 50000 points in a sphere of radius 8
            self._points_local = generate_points_in_sphere(50000, 8.0).to(device)

    @staticmethod
    def _v_pred_to_x0(noisy: torch.Tensor, v_pred: torch.Tensor, t_scalar: float, num_train_timestep: int) -> torch.Tensor:
        """FlowMatchEulerDiscrete convention: x_t = (1 - sigma) x0 + sigma * eps, v = eps - x0.
        Hence x0 = x_t - sigma * v with sigma = t / num_train_timestep."""
        sigma = float(t_scalar) / float(num_train_timestep)
        return noisy - sigma * v_pred

    def _make_timestep_vec(self, num_frames, value, device, dtype) -> torch.Tensor:
        return torch.full((num_frames,), float(value), device=device, dtype=dtype)

    @torch.no_grad()
    def _prefill_text(
        self,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        vision_states: torch.Tensor,
        byt5_text_states: torch.Tensor,
        byt5_text_mask: torch.Tensor,
        device,
        latent_dtype,
        mask_type: str = "i2v",
    ) -> List[Dict[str, Optional[torch.Tensor]]]:
        """One-shot forward to populate {k_txt, v_txt} for every double_block."""
        num_blocks = len(self.generator.double_blocks)
        kv_cache = _zero_kv_cache_list(num_blocks)
        t_expand_txt = torch.tensor([0], device=device, dtype=latent_dtype)
        extra_kwargs = {
            "byt5_text_states": byt5_text_states,
            "byt5_text_mask": byt5_text_mask,
        }
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            kv_cache = self.generator(
                bi_inference=False,
                ar_txt_inference=True,
                ar_vision_inference=False,
                timestep_txt=t_expand_txt,
                text_states=prompt_embeds,
                encoder_attention_mask=prompt_mask,
                vision_states=vision_states,
                mask_type=mask_type,
                extra_kwargs=extra_kwargs,
                kv_cache=kv_cache,
                cache_txt=True,
            )
        return kv_cache

    @torch.no_grad()
    def _prefill_memory(
        self,
        kv_cache,
        latents: torch.Tensor,           # already-generated [B, C, T, H, W] (history)
        cond_latents: torch.Tensor,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        action: torch.Tensor,
        chunk_i: int,
        device,
        target_dtype,
        mask_type: str = "i2v",
        noisy_memory: bool = False,
    ) -> Tuple[list, list]:
        """Pick memory frames from already-generated history and forward them
        with cache_vision=True to populate {k_vision, v_vision} for this chunk.

        If `noisy_memory` is True, sample a random stage1 scheduler index in
        [memory_noise_idx_min, memory_noise_idx_max) per source chunk, convert it
        to the true model timestep/sigma, and add fresh Gaussian noise to the
        context latents accordingly. This matches stage1 CameraJsonWMemDataset's
        select_window_out_flag==1 branch so the AR student stays in-distribution
        for heavily-noised memory KV.
        Otherwise (default), feeds clean x0_pred at stabilization_level-1,
        matching inference _ar_rollout_inner.
        """
        self._ensure_points_local(device)

        current_frame_idx = chunk_i * self.chunk_latent_frames
        selected_frame_indices = []
        for chunk_start_idx in range(
            current_frame_idx, current_frame_idx + self.chunk_latent_frames, 4
        ):
            selected_history_frame_id = select_aligned_memory_frames(
                viewmats[0].detach().to(torch.float32).cpu().numpy(),
                chunk_start_idx,
                memory_frames=self.memory_frames,
                temporal_context_size=self.temporal_context_size,
                pred_latent_size=self.pred_latent_size,
                points_local=self._points_local,
                device=device,
            )
            selected_frame_indices += selected_history_frame_id
        selected_frame_indices = sorted(list(set(selected_frame_indices)))
        to_remove = list(range(current_frame_idx, current_frame_idx + self.chunk_latent_frames))
        selected_frame_indices = [x for x in selected_frame_indices if x not in to_remove]

        if len(selected_frame_indices) == 0:
            return kv_cache, []

        # Important: detach history before feeding into prefill so the prefill
        # forward never propagates gradients back into earlier chunks.
        context_latents = latents[:, :, selected_frame_indices].detach()
        context_cond_latents = cond_latents[:, :, selected_frame_indices]
        context_viewmats = viewmats[:, selected_frame_indices].to(device)
        context_Ks = Ks[:, selected_frame_indices].to(device)
        context_action = action[:, selected_frame_indices].to(device)

        if noisy_memory:
            # NOTE: we deliberately do NOT use `self.scheduler.timesteps[idx]` /
            # `self.scheduler.sigmas[idx]` for the lookup. The training pipeline
            # calls `self.scheduler.set_timesteps(4)` during init, which
            # overwrites the scheduler's 1000-element default arrays with the
            # 4-step inference grid — so `self.scheduler.timesteps.numel() == 4`
            # by the time we get here, not 1000. Instead, replicate stage1's
            # exact mapping inline: stage1's noise_scheduler is
            # `diffusers.FlowMatchEulerDiscreteScheduler()` (default config,
            # never set_timesteps'd), where
            #     timesteps[i] = N - i,  sigmas[i] = (N - i) / N
            # for i in [0, N). So idx=500 -> t=500, sigma=0.5; idx=984 ->
            # t=16, sigma=0.016.
            N = int(self.num_train_timestep)
            if (
                self.memory_noise_idx_min < 0
                or self.memory_noise_idx_max <= self.memory_noise_idx_min
                or self.memory_noise_idx_max > N
            ):
                raise ValueError(
                    "Invalid memory-noise scheduler index range "
                    f"[{self.memory_noise_idx_min}, {self.memory_noise_idx_max}) "
                    f"for num_train_timestep {N}"
                )

            # Per-source-chunk random scheduler index (all frames from the same
            # source chunk share one index, matching stage1 line 322-324 which
            # assigns one rand_val per group of 4 indices). Sample on rank 0 and
            # broadcast so all SP ranks agree.
            sel_idx_t = torch.tensor(selected_frame_indices, device=device, dtype=torch.long)
            chunk_ids = sel_idx_t // self.chunk_latent_frames
            unique_chunks, inverse = torch.unique(chunk_ids, return_inverse=True)
            num_unique = int(unique_chunks.numel())
            per_chunk_idx = torch.empty(num_unique, device=device, dtype=torch.long)
            if (not dist.is_initialized()) or dist.get_rank() == 0:
                per_chunk_idx = torch.randint(
                    self.memory_noise_idx_min,
                    self.memory_noise_idx_max,
                    (num_unique,),
                    device=device,
                    dtype=torch.long,
                )
            per_chunk_idx = _broadcast_long_tensor(per_chunk_idx)
            per_frame_idx = per_chunk_idx[inverse]

            # Apply stage1's index -> (t, sigma) mapping inline.
            per_frame_t = (N - per_frame_idx).to(dtype=torch.float32)
            sigma = (per_frame_t / float(N)).view(1, 1, -1, 1, 1).to(
                dtype=context_latents.dtype
            )

            # Re-noise context latents at the matching sigma: same flow-match
            # forward-process as stage1 (noisy = (1-sigma)*latent + sigma*noise).
            # Sample noise on rank 0 and broadcast for SP determinism.
            fresh_noise = torch.randn_like(context_latents)
            if dist.is_initialized():
                dist.broadcast(fresh_noise, src=0)
            context_latents = (1.0 - sigma) * context_latents + sigma * fresh_noise
            context_timestep = per_frame_t
        else:
            context_timestep = torch.full(
                (len(selected_frame_indices),),
                self.stabilization_level - 1,
                device=device,
                dtype=torch.float32,  # match `_ar_rollout_inner` (timesteps.dtype)
            )

        context_latents_input = torch.concat(
            [context_latents, context_cond_latents], dim=1
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            kv_cache = self.generator(
                bi_inference=False,
                ar_txt_inference=False,
                ar_vision_inference=True,
                hidden_states=context_latents_input.to(target_dtype),
                timestep=context_timestep,
                timestep_r=None,
                mask_type=mask_type,
                return_dict=False,
                viewmats=context_viewmats.to(target_dtype),
                Ks=context_Ks.to(target_dtype),
                action=context_action.to(target_dtype),
                kv_cache=kv_cache,
                cache_vision=True,
                rope_temporal_size=context_latents_input.shape[2],
                start_rope_start_idx=0,
            )
        return kv_cache, selected_frame_indices

    def _denoise_chunk(
        self,
        kv_cache,
        noise_chunk: torch.Tensor,        # [B, C, chunk_T, H, W] starting noise
        cond_latents_chunk: torch.Tensor, # [B, C, chunk_T, H, W] i2v cond
        viewmats_chunk: torch.Tensor,
        Ks_chunk: torch.Tensor,
        action_chunk: torch.Tensor,
        num_history_frames: int,          # = len(selected_frame_indices)
        exit_step_idx: int,
        requires_grad: bool,
        device,
        target_dtype,
        mask_type: str = "i2v",
    ) -> torch.Tensor:
        """Run num_inference_steps denoising on a single chunk.

        At step `exit_step_idx`, model is invoked under torch.enable_grad()
        (when `requires_grad`) and its x0 prediction is returned. Earlier
        steps run in no_grad and Euler-step to the next iteration — matching
        HY `_ar_rollout_inner` (`scheduler.step`), which never re-noises."""
        chunk_T = noise_chunk.shape[2]
        x_t = noise_chunk
        x0_pred = None
        for step_idx, t_scalar in enumerate(self.denoising_step_list):
            # float32 timestep to match `_ar_rollout_inner` (its `timestep_input`
            # uses `timesteps.dtype`, which is float32). bf16 would round few-step
            # values like 937.5 -> 936, so the model would see different timesteps
            # in training vs inference.
            timestep_input = self._make_timestep_vec(
                chunk_T, t_scalar, device, torch.float32
            )
            x_cat = torch.concat([x_t, cond_latents_chunk], dim=1).to(target_dtype)
            x_cat_scaled = self.scheduler.scale_model_input(x_cat, t_scalar)

            grad_on = requires_grad and step_idx == exit_step_idx
            ctx = torch.enable_grad() if grad_on else torch.no_grad()
            with ctx, torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                v_pred = self.generator(
                    bi_inference=False,
                    ar_txt_inference=False,
                    ar_vision_inference=True,
                    hidden_states=x_cat_scaled,
                    timestep=timestep_input,
                    timestep_r=None,
                    mask_type=mask_type,
                    return_dict=False,
                    viewmats=viewmats_chunk.to(target_dtype),
                    Ks=Ks_chunk.to(target_dtype),
                    action=action_chunk.to(target_dtype),
                    kv_cache=kv_cache,
                    cache_vision=False,
                    rope_temporal_size=x_cat.shape[2] + num_history_frames,
                    start_rope_start_idx=num_history_frames,
                )[0]

            x0_pred = self._v_pred_to_x0(x_t, v_pred, t_scalar, self.num_train_timestep)

            if step_idx == exit_step_idx:
                break

            # Advance to the next denoising step with the SAME deterministic
            # Euler update HY inference uses — `FlowMatchDiscreteScheduler.step`
            # inside `_ar_rollout_inner`: prev = sample + v * (sigma_next - sigma),
            # computed in fp32 then stored back at the latent dtype. HY's 4-step
            # rollout NEVER re-noises (verified: run_ar_4step.sh and
            # run_ar_50step.sh both go through `_ar_rollout_inner`, whose only
            # per-step transition is this Euler step). The training rollout must
            # match inference byte-for-byte, so we Euler-step here too — NO fresh
            # re-noising.
            if step_idx < len(self.denoising_step_list) - 1:
                next_t = self.denoising_step_list[step_idx + 1]
                sigma_cur = float(t_scalar) / float(self.num_train_timestep)
                sigma_next = float(next_t) / float(self.num_train_timestep)
                x_t = (x_t.float() + v_pred.float() * (sigma_next - sigma_cur)).to(
                    noise_chunk.dtype
                )

        return x0_pred

    def rollout(
        self,
        *,
        initial_latent: torch.Tensor,            # [B, C, T_full, H, W] — full latent canvas (random init)
        cond_latents: torch.Tensor,              # [B, C, T_full, H, W]
        viewmats: torch.Tensor,                  # [B, T_full, 4, 4]
        Ks: torch.Tensor,                        # [B, T_full, 3, 3]
        action: torch.Tensor,                    # [B, T_full]
        prompt_embeds: torch.Tensor,             # [B, S, D]
        prompt_mask: torch.Tensor,               # [B, S]
        vision_states: torch.Tensor,             # [B, V, D_v]
        byt5_text_states: torch.Tensor,
        byt5_text_mask: torch.Tensor,
        target_dtype: torch.dtype = torch.bfloat16,
        mask_type: str = "i2v",
        inference_mode: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """Run self-forcing rollout. Returns:
            clean_latents [B, C, T_full, H, W]
            gradient_mask [B, T_full] (bool) — True for frames whose x0_pred is
              connected to generator gradients.
            denoised_timestep_from, denoised_timestep_to — global aggregate
              "noise band" of the rollout's exit steps, useful for DMD ts sched.
        """
        device = initial_latent.device
        B, C, T_full, H, W = initial_latent.shape
        assert T_full % self.chunk_latent_frames == 0, (
            f"T_full={T_full} not divisible by chunk_latent_frames={self.chunk_latent_frames}"
        )
        num_chunks = T_full // self.chunk_latent_frames

        # --- Step 1: text/byt5/siglip prefill ---
        kv_cache = self._prefill_text(
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            vision_states=vision_states,
            byt5_text_states=byt5_text_states,
            byt5_text_mask=byt5_text_mask,
            device=device,
            latent_dtype=initial_latent.dtype,
            mask_type=mask_type,
        )

        # --- Step 2: exit-step per chunk ---
        # inference_mode (validation): deterministic — every chunk runs the full
        # denoise (exit at the last step) and NO chunk carries gradient.
        if inference_mode:
            exit_steps = [len(self.denoising_step_list) - 1] * num_chunks
        else:
            exit_steps = _broadcast_exit_step(
                num_chunks, len(self.denoising_step_list), device
            )

        # --- Step 2b: stage1-parity memory-noise toggle (per-rollout) ---
        # With prob `memory_noise_prob`, all _prefill_memory calls in this
        # rollout re-noise context latents at a random heavy timestep — matches
        # stage1 CameraJsonWMemDataset select_window_out_flag==1 (~80%). Force
        # OFF in inference_mode so validation rollouts mirror generate.py.
        if inference_mode or self.memory_noise_prob <= 0.0:
            noisy_memory = False
        else:
            rank0_flag = torch.rand((), device=device).item() < self.memory_noise_prob
            noisy_memory = _broadcast_bool(rank0_flag, device)

        # Compute per-chunk grad mask. inference_mode -> no grad anywhere.
        # Else if `slice_last_frames` is None, all chunks have grad (LongLive
        # train_init); otherwise only the last `slice_last_frames` latents have
        # grad (LongLive train_long).
        if inference_mode:
            first_grad_chunk = num_chunks
        elif self.slice_last_frames is None:
            first_grad_chunk = 0
        else:
            grad_chunks = max(1, self.slice_last_frames // self.chunk_latent_frames)
            first_grad_chunk = max(0, num_chunks - grad_chunks)

        # --- Step 3: chunk-by-chunk causal denoising ---
        latents_full = initial_latent.clone()
        grad_mask = torch.zeros((B, T_full), dtype=torch.bool, device=device)
        for chunk_i in range(num_chunks):
            start_idx = chunk_i * self.chunk_latent_frames
            end_idx = start_idx + self.chunk_latent_frames

            # 3a) memory-frame prefill (chunk_i > 0)
            num_history = 0
            if chunk_i > 0:
                kv_cache, selected_idx = self._prefill_memory(
                    kv_cache=kv_cache,
                    latents=latents_full,
                    cond_latents=cond_latents,
                    viewmats=viewmats,
                    Ks=Ks,
                    action=action,
                    chunk_i=chunk_i,
                    device=device,
                    target_dtype=target_dtype,
                    mask_type=mask_type,
                    noisy_memory=noisy_memory,
                )
                num_history = len(selected_idx)

            # 3b) denoise current chunk
            noise_chunk = latents_full[:, :, start_idx:end_idx]
            cond_chunk = cond_latents[:, :, start_idx:end_idx]
            view_chunk = viewmats[:, start_idx:end_idx]
            Ks_chunk = Ks[:, start_idx:end_idx]
            act_chunk = action[:, start_idx:end_idx]

            chunk_requires_grad = chunk_i >= first_grad_chunk

            # Per-chunk gradient checkpointing: wrap the chunk's denoise forward
            # in torch.utils.checkpoint so cross-chunk activations get freed
            # after forward and recomputed during backward. Only useful when this
            # chunk has grad; no-grad chunks already save nothing.
            # Default OFF — empirically didn't help (forward peak dominates).
            use_per_chunk_ckpt = chunk_requires_grad and bool(
                int(os.environ.get("DMD_PER_CHUNK_CHECKPOINT", "0"))
            )
            if use_per_chunk_ckpt:
                from torch.utils.checkpoint import checkpoint

                def _chunk_fn(noise_in, cond_in, view_in, Ks_in, act_in):
                    return self._denoise_chunk(
                        kv_cache=kv_cache,
                        noise_chunk=noise_in,
                        cond_latents_chunk=cond_in,
                        viewmats_chunk=view_in,
                        Ks_chunk=Ks_in,
                        action_chunk=act_in,
                        num_history_frames=num_history,
                        exit_step_idx=exit_steps[chunk_i],
                        requires_grad=True,
                        device=device,
                        target_dtype=target_dtype,
                        mask_type=mask_type,
                    )

                x0_pred = checkpoint(
                    _chunk_fn,
                    noise_chunk, cond_chunk, view_chunk, Ks_chunk, act_chunk,
                    use_reentrant=False,
                )
            else:
                x0_pred = self._denoise_chunk(
                    kv_cache=kv_cache,
                    noise_chunk=noise_chunk,
                    cond_latents_chunk=cond_chunk,
                    viewmats_chunk=view_chunk,
                    Ks_chunk=Ks_chunk,
                    action_chunk=act_chunk,
                    num_history_frames=num_history,
                    exit_step_idx=exit_steps[chunk_i],
                    requires_grad=chunk_requires_grad,
                    device=device,
                    target_dtype=target_dtype,
                    mask_type=mask_type,
                )

            # Drop the cond_latents half that was packed into the model input.
            x0_only = x0_pred[:, :, -self.chunk_latent_frames:]
            # Detach when this chunk has no grad so latents_full stays free of
            # autograd refs from no_grad chunks.
            if not chunk_requires_grad:
                x0_only = x0_only.detach()
            latents_full[:, :, start_idx:end_idx] = x0_only
            grad_mask[:, start_idx:end_idx] = chunk_requires_grad

        # Aggregate exit-step band (logged as ts_from / ts_to; also used by
        # WorldPlayDMD when ts_schedule=True). `denoising_step_list` is
        # DESCENDING (high t -> low t), so the SMALLEST exit index is the
        # HIGHEST-noise timestep. `from` = high-noise end, `to` = low-noise end.
        # The old code assigned these inverted (from <- min t, to <- max t),
        # which made `_re_noise`'s [to, from] band degenerate and pinned t at
        # ~999 every step.
        denoising_step_list_t = torch.tensor(self.denoising_step_list)
        denoised_timestep_from = int(denoising_step_list_t[min(exit_steps)].item())
        denoised_timestep_to = int(denoising_step_list_t[max(exit_steps)].item())
        return latents_full, grad_mask, denoised_timestep_from, denoised_timestep_to
