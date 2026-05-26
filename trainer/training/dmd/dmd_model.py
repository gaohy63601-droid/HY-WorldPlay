# SPDX-License-Identifier: Apache-2.0
"""
DMD model wrapper for WorldPlay distillation.

Holds three transformers:
  * generator   — AR causal student, KV-cache enabled, trainable.
                  Initialized from HY-World1.5-Autoregressive-480P-I2V.
  * real_score  — bidirectional teacher, frozen.
                  Initialized from HY-World1.5-Bidirectional-480P-I2V.
  * fake_score  — bidirectional critic, trainable. Cloned from real_score.

Generator path runs `WorldPlaySelfForcingPipeline.rollout(...)` to get clean
self-rollout latents, then DMD computes
    grad = (pred_fake - pred_real) / normalizer
on a randomly-sampled re-noise timestep. The student backward is wrapped in a
surrogate MSE so PyTorch sees a scalar loss whose backward is equivalent to
injecting `-grad` as an external gradient on the clean rollout latents.

Critic path runs the same rollout under `torch.no_grad()`, re-noises and
predicts x0 via fake_score, then trains fake_score with a flow-matching
denoising loss (target: noise minus clean, i.e. velocity).
"""

from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def _get_timestep_band(
    min_t: int,
    max_t: int,
    batch_size: int,
    num_frame: int,
    num_frame_per_block: int,
    device,
    uniform_per_block: bool = True,
) -> torch.Tensor:
    """Uniformly sample timesteps in [min_t, max_t). Optionally constant within
    each block of `num_frame_per_block` frames (matches LongLive)."""
    if uniform_per_block:
        # One scalar per batch, broadcast across frames.
        t = torch.randint(min_t, max_t, (batch_size, 1), device=device, dtype=torch.long)
        return t.repeat(1, num_frame)
    t = torch.randint(min_t, max_t, (batch_size, num_frame), device=device, dtype=torch.long)
    # Snap to per-block constant
    t = t.reshape(batch_size, -1, num_frame_per_block)
    t[:, :, 1:] = t[:, :, 0:1]
    return t.reshape(batch_size, -1)


class WorldPlayDMD(nn.Module):
    """Compute generator and critic losses in a single self-contained module.

    The actual transformer instances are passed in (already constructed +
    state-dict-loaded). FSDP wrapping happens *outside* this class to keep
    sharding orthogonal to loss computation.
    """

    def __init__(
        self,
        generator,
        real_score,
        fake_score,
        scheduler,
        self_forcing_pipeline,           # WorldPlaySelfForcingPipeline
        *,
        num_train_timestep: int = 1000,
        denoising_step_list,  # required: derived from the shift-matched scheduler by the caller
        real_guidance_scale: float = 6.0,
        fake_guidance_scale: float = 0.0,
        min_score_ratio: float = 0.02,
        max_score_ratio: float = 0.98,
        ts_schedule: bool = True,
        ts_schedule_max: bool = False,
        # SD3 timestep shift applied to the DMD re-noise timestep — must equal
        # the noise-schedule shift (LongLive train_init: timestep_shift=5.0).
        timestep_shift: float = 5.0,
        min_score_timestep: int = 0,
        chunk_latent_frames: int = 4,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.generator = generator
        self.real_score = real_score
        self.fake_score = fake_score
        self.scheduler = scheduler
        self.pipe = self_forcing_pipeline

        self.num_train_timestep = num_train_timestep
        self.denoising_step_list = list(denoising_step_list)
        self.real_guidance_scale = real_guidance_scale
        self.fake_guidance_scale = fake_guidance_scale
        self.min_step = int(min_score_ratio * num_train_timestep)
        self.max_step = int(max_score_ratio * num_train_timestep)
        self.ts_schedule = ts_schedule
        self.ts_schedule_max = ts_schedule_max
        self.timestep_shift = timestep_shift
        self.min_score_timestep = min_score_timestep
        self.chunk_latent_frames = chunk_latent_frames
        self.device = device

        # Real teacher is frozen.
        for p in self.real_score.parameters():
            p.requires_grad_(False)
        self.real_score.eval()

    # ---------------------------------------------------------------- helpers

    def _bi_forward(
        self,
        score_model,
        noisy_latents: torch.Tensor,
        cond_latents: torch.Tensor,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        action: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        vision_states: torch.Tensor,
        byt5_text_states: torch.Tensor,
        byt5_text_mask: torch.Tensor,
        timestep: torch.Tensor,
        target_dtype,
        mask_type: str = "i2v",
    ) -> torch.Tensor:
        """One-shot bidirectional forward through real/fake score model.
        Returns v_pred (velocity prediction in flow-match parameterization)."""
        latents_concat = torch.concat([noisy_latents, cond_latents], dim=1).to(target_dtype)
        latents_concat = self.scheduler.scale_model_input(latents_concat, float(timestep[0].item()))
        extra_kwargs = {
            "byt5_text_states": byt5_text_states,
            "byt5_text_mask": byt5_text_mask,
        }
        t_expand_txt = torch.zeros((1,), device=noisy_latents.device, dtype=noisy_latents.dtype)

        # forward_bi expects flat [B*T] for both `timestep` and `action`
        # (time_in / action_in -> timestep_embedding which calls t[:, None] * freqs[None]).
        B = noisy_latents.shape[0]
        T_latent = noisy_latents.shape[2]
        if timestep.dim() == 1 and timestep.shape[0] == B:
            timestep_flat = timestep.unsqueeze(1).expand(B, T_latent).reshape(-1)
        else:
            timestep_flat = timestep.reshape(-1)
        action_flat = action.to(target_dtype).reshape(-1)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            v_pred, _ = score_model(
                bi_inference=True,
                hidden_states=latents_concat,
                timestep=timestep_flat,
                timestep_txt=t_expand_txt,
                text_states=prompt_embeds,
                text_states_2=None,
                encoder_attention_mask=prompt_mask,
                vision_states=vision_states,
                mask_type=mask_type,
                extra_kwargs=extra_kwargs,
                action=action_flat,
                viewmats=viewmats.to(target_dtype),
                Ks=Ks.to(target_dtype),
                return_dict=False,
            )
        return v_pred

    @staticmethod
    def _v_to_x0(noisy: torch.Tensor, v_pred: torch.Tensor, t: torch.Tensor, num_train_timestep: int) -> torch.Tensor:
        # sigma = t / num_train_timestep, broadcast over latent dims
        sigma = (t.float() / float(num_train_timestep)).view(noisy.shape[0], 1, 1, 1, 1)
        return noisy - sigma * v_pred

    # --------------------------------------------------- KL / DMD computation

    def _compute_kl_grad(
        self,
        noisy_latents: torch.Tensor,
        clean_estimate: torch.Tensor,
        timestep: torch.Tensor,
        cond_kwargs: Dict,
        uncond_kwargs: Dict,
        target_dtype,
    ) -> Tuple[torch.Tensor, Dict]:
        """grad = pred_fake - pred_real (both predicting x0). Optional gradient
        normalization by |x0 - pred_real| to stabilize early training."""
        # ---- fake score (no CFG by default) ----
        v_fake_c = self._bi_forward(
            self.fake_score, noisy_latents, timestep=timestep, target_dtype=target_dtype, **cond_kwargs
        )
        x0_fake = self._v_to_x0(noisy_latents, v_fake_c, timestep, self.num_train_timestep)
        if self.fake_guidance_scale != 0.0:
            v_fake_u = self._bi_forward(
                self.fake_score, noisy_latents, timestep=timestep, target_dtype=target_dtype, **uncond_kwargs
            )
            x0_fake_u = self._v_to_x0(noisy_latents, v_fake_u, timestep, self.num_train_timestep)
            x0_fake = x0_fake + (x0_fake - x0_fake_u) * self.fake_guidance_scale

        # ---- real score (CFG) ----
        v_real_c = self._bi_forward(
            self.real_score, noisy_latents, timestep=timestep, target_dtype=target_dtype, **cond_kwargs
        )
        v_real_u = self._bi_forward(
            self.real_score, noisy_latents, timestep=timestep, target_dtype=target_dtype, **uncond_kwargs
        )
        x0_real_c = self._v_to_x0(noisy_latents, v_real_c, timestep, self.num_train_timestep)
        x0_real_u = self._v_to_x0(noisy_latents, v_real_u, timestep, self.num_train_timestep)
        x0_real = x0_real_c + (x0_real_c - x0_real_u) * self.real_guidance_scale

        grad = x0_fake - x0_real
        # Per-sample normalizer = mean(|clean_estimate - x0_real|) over (C,T,H,W)
        normalizer = torch.abs(clean_estimate - x0_real).mean(dim=(1, 2, 3, 4), keepdim=True)
        grad = grad / normalizer.clamp_min(1e-6)
        grad = torch.nan_to_num(grad)
        return grad, {
            "dmd_gradient_abs_mean": torch.mean(torch.abs(grad)).detach(),
            "dmd_normalizer_mean": normalizer.mean().detach(),
        }

    def _re_noise(
        self,
        clean: torch.Tensor,
        denoised_timestep_from: int,
        denoised_timestep_to: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Timestep sampling mirrors LongLive `model/dmd.py`
        # (_compute_distribution_matching_loss / critic): uniform-sample in
        # [min_t, max_t] — with ts_schedule(/_max) off that is
        # [min_score_timestep, num_train_timestep] = [0, 1000] — then apply the
        # SD3 timestep shift (same `shift` as the noise schedule), then clamp to
        # [min_step, max_step] (~[20, 980]). The shift skews the score-matching
        # timesteps toward high noise to match where a shift-5 few-step model
        # operates; a plain uniform [20, 980] would over-weight low noise.
        B, C, T, H, W = clean.shape
        min_t = int(denoised_timestep_to) if self.ts_schedule else self.min_score_timestep
        max_t = int(denoised_timestep_from) if self.ts_schedule_max else self.num_train_timestep
        if max_t <= min_t:
            max_t = min_t + 1
        timestep = _get_timestep_band(
            min_t, max_t, B, T, self.chunk_latent_frames, clean.device, uniform_per_block=True
        ).float()
        # SD3 timestep shift — identical formula to
        # FlowMatchDiscreteScheduler.sd3_time_shift.
        if self.timestep_shift > 1.0:
            r = timestep / float(self.num_train_timestep)
            timestep = (
                self.timestep_shift * r
                / (1.0 + (self.timestep_shift - 1.0) * r)
                * float(self.num_train_timestep)
            )
        timestep = timestep.clamp(self.min_step, self.max_step)
        # Use just one scalar timestep per batch for the BI score forward (frame-uniform).
        t_scalar = timestep[:, 0]
        noise = torch.randn_like(clean)
        # Flow matching: x_t = (1 - sigma) * x0 + sigma * eps, sigma = t / T.
        sigma = (t_scalar.float() / float(self.num_train_timestep)).view(B, 1, 1, 1, 1)
        noisy = (1.0 - sigma) * clean + sigma * noise
        return noisy, noise, t_scalar.to(torch.long)

    # ----------------------------------------------------- public loss methods

    def generator_loss(
        self,
        batch: Dict,
        target_dtype: torch.dtype = torch.bfloat16,
    ) -> Tuple[torch.Tensor, Dict]:
        """Run self-forcing rollout (grad enabled on one denoising step per
        chunk) and apply DMD distribution-matching loss on the clean output."""
        clean, grad_mask, ts_from, ts_to = self.pipe.rollout(
            initial_latent=batch["initial_noise"],
            cond_latents=batch["cond_latents"],
            viewmats=batch["viewmats"],
            Ks=batch["Ks"],
            action=batch["action"],
            prompt_embeds=batch["prompt_embeds"],
            prompt_mask=batch["prompt_mask"],
            vision_states=batch["vision_states"],
            byt5_text_states=batch["byt5_text_states"],
            byt5_text_mask=batch["byt5_text_mask"],
            target_dtype=target_dtype,
        )

        cond_kwargs = dict(
            cond_latents=batch["cond_latents"],
            viewmats=batch["viewmats"],
            Ks=batch["Ks"],
            action=batch["action"],
            prompt_embeds=batch["prompt_embeds"],
            prompt_mask=batch["prompt_mask"],
            vision_states=batch["vision_states"],
            byt5_text_states=batch["byt5_text_states"],
            byt5_text_mask=batch["byt5_text_mask"],
        )
        uncond_kwargs = dict(
            cond_latents=batch["cond_latents"],
            viewmats=batch["viewmats"],
            Ks=batch["Ks"],
            action=batch["action"],
            prompt_embeds=batch["uncond_prompt_embeds"],
            prompt_mask=batch["uncond_prompt_mask"],
            vision_states=batch["uncond_vision_states"],
            byt5_text_states=batch["uncond_byt5_text_states"],
            byt5_text_mask=batch["uncond_byt5_text_mask"],
        )

        with torch.no_grad():
            noisy, _, ts = self._re_noise(clean.detach(), ts_from, ts_to)
            grad, log = self._compute_kl_grad(
                noisy_latents=noisy,
                clean_estimate=clean.detach(),
                timestep=ts,
                cond_kwargs=cond_kwargs,
                uncond_kwargs=uncond_kwargs,
                target_dtype=target_dtype,
            )

        # surrogate MSE — backward through `clean` equivalent to injecting -grad
        loss = 0.5 * F.mse_loss(
            clean.float(), (clean.float() - grad.float()).detach(), reduction="mean"
        )
        return loss, {
            "ts_from": ts_from,
            "ts_to": ts_to,
            "rollout_clean_abs_mean": clean.detach().abs().mean(),
            **log,
        }

    def critic_loss(
        self,
        batch: Dict,
        target_dtype: torch.dtype = torch.bfloat16,
    ) -> Tuple[torch.Tensor, Dict]:
        with torch.no_grad():
            clean, _, ts_from, ts_to = self.pipe.rollout(
                initial_latent=batch["initial_noise"],
                cond_latents=batch["cond_latents"],
                viewmats=batch["viewmats"],
                Ks=batch["Ks"],
                action=batch["action"],
                prompt_embeds=batch["prompt_embeds"],
                prompt_mask=batch["prompt_mask"],
                vision_states=batch["vision_states"],
                byt5_text_states=batch["byt5_text_states"],
                byt5_text_mask=batch["byt5_text_mask"],
                target_dtype=target_dtype,
            )
            noisy, noise, ts = self._re_noise(clean, ts_from, ts_to)

        cond_kwargs = dict(
            cond_latents=batch["cond_latents"],
            viewmats=batch["viewmats"],
            Ks=batch["Ks"],
            action=batch["action"],
            prompt_embeds=batch["prompt_embeds"],
            prompt_mask=batch["prompt_mask"],
            vision_states=batch["vision_states"],
            byt5_text_states=batch["byt5_text_states"],
            byt5_text_mask=batch["byt5_text_mask"],
        )
        # Train fake_score with flow-match denoising target: target velocity = (noise - clean)
        v_pred = self._bi_forward(
            self.fake_score, noisy, timestep=ts, target_dtype=target_dtype, **cond_kwargs
        )
        target_v = (noise - clean).to(v_pred.dtype)
        loss = F.mse_loss(v_pred, target_v, reduction="mean")
        return loss, {
            "critic_ts_mean": ts.float().mean(),
            "critic_target_abs_mean": target_v.abs().mean(),
        }
