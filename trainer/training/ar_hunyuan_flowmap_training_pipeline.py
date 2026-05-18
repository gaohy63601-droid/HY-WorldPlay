# SPDX-License-Identifier: Apache-2.0
import os
import sys

sys.path.append(os.path.abspath("."))

import math

import torch
import torch.distributed as dist
from einops import rearrange

from trainer.distributed import get_local_torch_device, get_sp_group
from trainer.forward_context import set_forward_context
from trainer.logger import init_logger
from trainer.models.hyvideo.models.transformers.modules.activation_layers import (
    get_activation_layer,
)
from trainer.models.hyvideo.models.transformers.modules.embed_layers import (
    TimestepEmbedder,
)
from trainer.trainer_args import TrainerArgs, TrainingArgs
from trainer.training.ar_hunyuan_mem_training_pipeline import (
    TrainingPipeline,
    vsa_available,
)
from trainer.training.training_utils import compute_density_for_timestep_sampling
from trainer.utils import FlexibleArgumentParser, is_vsa_available

logger = init_logger(__name__)


class FlowMapHunyuanTrainingPipeline(TrainingPipeline):
    """
    Flow-map causal student training for WorldPlay.

    This keeps the original AR/memory dataloader and conditioning path, but
    trains the current causal student with AnyFlow-style (t, r) transition
    targets. The ordinary causal student pipeline remains unchanged.
    """

    _required_config_modules = ["transformer"]

    def initialize_pipeline(self, trainer_args: TrainerArgs):
        pass

    def create_training_stages(self, training_args: TrainingArgs):
        pass

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        pass

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        super().initialize_training_pipeline(training_args)
        self._setup_flowmap_time_embedding()
        logger.info(
            "Flow-map training enabled: diffusion_ratio=%s, consistency_ratio=%s, epsilon=%s",
            training_args.flowmap_diffusion_ratio,
            training_args.flowmap_consistency_ratio,
            training_args.flowmap_epsilon,
        )

    def _setup_flowmap_time_embedding(self) -> None:
        module = self.transformer
        base = module
        if hasattr(module, "module"):
            base = module.module

        if getattr(base, "time_r_in", None) is None:
            raise RuntimeError(
                "Flow-map r-timestep embedding was not initialized before FSDP. "
                "Check trainer.models.loader.fsdp_load flowmap_training setup."
            )

        base.flowmap_gate_value = self.training_args.flowmap_gate_value
        base.flowmap_deltatime_type = self.training_args.flowmap_deltatime_type

        if hasattr(base, "register_to_config"):
            base.register_to_config(use_meanflow=True)

    def _sample_flowmap_timesteps(self, batch_size, latent_t, device, dtype):
        chunk_latent_num = 4
        num_chunks = math.ceil(latent_t / chunk_latent_num)
        total_chunks = batch_size * num_chunks
        u_t = compute_density_for_timestep_sampling(
            weighting_scheme=self.training_args.weighting_scheme,
            batch_size=total_chunks,
            generator=self.noise_random_generator,
            logit_mean=self.training_args.logit_mean,
            logit_std=self.training_args.logit_std,
            mode_scale=self.training_args.mode_scale,
        ).to(device=device, dtype=dtype).reshape(batch_size, num_chunks)
        u_r = torch.rand(
            (batch_size, num_chunks),
            generator=self.noise_gen_cuda,
            device=device,
            dtype=dtype,
        )
        t_u = torch.maximum(u_t, u_r)
        r_u = torch.minimum(u_t, u_r)

        global_rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        global_start = global_rank * total_chunks
        total_global = world_size * total_chunks
        n_diffusion = round(self.training_args.flowmap_diffusion_ratio * total_global)
        n_consistency = round(self.training_args.flowmap_consistency_ratio * total_global)
        global_indices = torch.arange(
            global_start, global_start + total_chunks, device=device, dtype=torch.long
        ).reshape(batch_size, num_chunks)
        diffusion_mask = global_indices < n_diffusion
        consistency_mask = (global_indices >= n_diffusion) & (
            global_indices < n_diffusion + n_consistency
        )
        r_u = torch.where(diffusion_mask, t_u, r_u)
        r_u = torch.where(consistency_mask, torch.zeros_like(r_u), r_u)

        t_idx = (t_u * self.noise_scheduler.config.num_train_timesteps).long()
        r_idx = (r_u * self.noise_scheduler.config.num_train_timesteps).long()
        t_idx = (
            self.noise_scheduler.config.num_train_timesteps
            - self.timestep_transform(t_idx, self.train_time_shift)
        ).long()
        r_idx = (
            self.noise_scheduler.config.num_train_timesteps
            - self.timestep_transform(r_idx, self.train_time_shift)
        ).long()
        t_idx = t_idx.clamp(0, len(self.noise_scheduler.timesteps) - 1)
        r_idx = r_idx.clamp(0, len(self.noise_scheduler.timesteps) - 1)

        t_idx = t_idx.repeat_interleave(chunk_latent_num, dim=1)[:, :latent_t]
        r_idx = r_idx.repeat_interleave(chunk_latent_num, dim=1)[:, :latent_t]

        scheduler_timesteps = self.noise_scheduler.timesteps.to(device=device)
        timesteps = scheduler_timesteps[t_idx.reshape(-1)]
        r_timesteps = scheduler_timesteps[r_idx.reshape(-1)]

        if self.training_args.sp_size > 1:
            sp_group = get_sp_group()
            sp_group.broadcast(timesteps, src=0)
            sp_group.broadcast(r_timesteps, src=0)

        return timesteps, r_timesteps

    def _apply_outside_window_timesteps(self, training_batch):
        if training_batch.select_window_out_flag != 1:
            return training_batch

        latent_t = training_batch.latents.shape[2]
        if latent_t <= 4:
            return training_batch

        timesteps = training_batch.timesteps.reshape(-1).reshape(-1, latent_t)
        r_timesteps = training_batch.r_timesteps.reshape(-1).reshape(-1, latent_t)
        scheduler_timesteps = self.noise_scheduler.timesteps.to(device=timesteps.device)
        for batch_idx in range(timesteps.shape[0]):
            for start in range(0, latent_t - 4, 4):
                rand_idx = torch.randint(500, 985, (1,), device=timesteps.device).clamp(
                    0, len(scheduler_timesteps) - 1
                )
                forced_timestep = scheduler_timesteps[rand_idx].to(timesteps.dtype)
                timesteps[batch_idx, start:start + 4] = forced_timestep
                r_timesteps[batch_idx, start:start + 4] = forced_timestep

        training_batch.timesteps = timesteps.reshape(-1)
        training_batch.r_timesteps = r_timesteps.reshape(-1)
        return training_batch

    def _prepare_ar_dit_inputs(self, training_batch):
        latents = training_batch.latents
        batch_size, _, latent_t, _, _ = latents.shape
        noise = torch.randn(
            latents.shape,
            generator=self.noise_gen_cuda,
            device=latents.device,
            dtype=latents.dtype,
        )

        timesteps, r_timesteps = self._sample_flowmap_timesteps(
            batch_size=batch_size,
            latent_t=latent_t,
            device=latents.device,
            dtype=latents.dtype,
        )
        training_batch.timesteps = timesteps
        training_batch.r_timesteps = r_timesteps
        training_batch = self._apply_outside_window_timesteps(training_batch)
        timesteps = training_batch.timesteps
        r_timesteps = training_batch.r_timesteps

        sigmas = self._sigmas_from_timesteps(timesteps, latents)
        noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise

        training_batch.noisy_model_input = noisy_model_input
        training_batch.sigmas = sigmas
        training_batch.noise = noise
        training_batch.raw_latent_shape = latents.shape
        return training_batch

    def _sigmas_from_timesteps(self, timesteps, latents):
        schedule_timesteps = self.noise_scheduler.timesteps.to(
            device=latents.device, dtype=timesteps.dtype
        )
        timestep_ids = [
            (schedule_timesteps == t).nonzero().item() for t in timesteps
        ]
        sigma = self.noise_scheduler.sigmas.to(
            device=latents.device, dtype=latents.dtype
        )[timestep_ids]
        sigma = sigma.reshape(-1, 1, 1, 1, 1)
        return rearrange(sigma, "(B D) C T H W -> B C (D T) H W", D=latents.shape[2])

    def _build_input_kwargs(self, training_batch):
        super()._build_input_kwargs(training_batch)
        training_batch.input_kwargs["timestep_r"] = training_batch.r_timesteps.to(
            get_local_torch_device(), dtype=torch.bfloat16
        )
        return training_batch

    def _flowmap_forward(self, training_batch, hidden_states, timesteps, r_timesteps):
        old_hidden_states = training_batch.input_kwargs["hidden_states"]
        old_timesteps = training_batch.input_kwargs["timestep"]
        old_r_timesteps = training_batch.input_kwargs["timestep_r"]
        training_batch.input_kwargs["hidden_states"] = hidden_states
        training_batch.input_kwargs["timestep"] = timesteps.to(
            get_local_torch_device(), dtype=torch.bfloat16
        )
        training_batch.input_kwargs["timestep_r"] = r_timesteps.to(
            get_local_torch_device(), dtype=torch.bfloat16
        )
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                return self.transformer(**training_batch.input_kwargs)[0]
        finally:
            training_batch.input_kwargs["hidden_states"] = old_hidden_states
            training_batch.input_kwargs["timestep"] = old_timesteps
            training_batch.input_kwargs["timestep_r"] = old_r_timesteps

    def _debug_stats(self, name, tensor):
        tensor = tensor.detach().float()
        return {
            f"flowmap_debug/{name}_abs_mean": tensor.abs().mean().item(),
            f"flowmap_debug/{name}_rms": torch.sqrt(torch.mean(tensor * tensor)).item(),
            f"flowmap_debug/{name}_abs_max": tensor.abs().max().item(),
        }

    def _record_flowmap_debug_metrics(
        self,
        training_batch,
        model_pred,
        target,
        v_pred,
        dF_dt,
        t_minus_r,
        i2v_mask,
        diff,
    ):
        metrics = {}
        metrics.update(self._debug_stats("model_pred", model_pred))
        metrics.update(self._debug_stats("target", target))
        metrics.update(self._debug_stats("v_pred", v_pred))
        metrics.update(self._debug_stats("dF_dt", dF_dt))
        metrics.update(self._debug_stats("t_minus_r", t_minus_r))

        masked_model_pred = model_pred.float() * i2v_mask
        masked_target = target.float() * i2v_mask
        metrics.update(self._debug_stats("masked_model_pred", masked_model_pred))
        metrics.update(self._debug_stats("masked_target", masked_target))
        metrics["flowmap_debug/mask_sum"] = float(i2v_mask.sum().item())
        metrics["flowmap_debug/diff_abs_mean"] = diff.detach().float().abs().mean().item()
        metrics["flowmap_debug/timestep_abs_mean"] = training_batch.timesteps.detach().float().abs().mean().item()
        metrics["flowmap_debug/r_timestep_abs_mean"] = training_batch.r_timesteps.detach().float().abs().mean().item()

        dist_metrics = {}
        for key, value in metrics.items():
            value_tensor = torch.tensor(value, device=get_local_torch_device(), dtype=torch.float32)
            dist.all_reduce(value_tensor, op=dist.ReduceOp.MAX)
            dist_metrics[key] = value_tensor.item()

        training_batch.flowmap_debug_metrics = dist_metrics
        if self.global_rank == 0:
            logger.info("Flow-map debug metrics: %s", dist_metrics)

    @torch.no_grad()
    def _compute_central_difference(self, training_batch, v_pred):
        epsilon = self.training_args.flowmap_epsilon
        num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
        delta = epsilon / num_train_timesteps

        hidden_plus = training_batch.input_kwargs["hidden_states"].clone()
        hidden_minus = training_batch.input_kwargs["hidden_states"].clone()
        hidden_plus[:, : v_pred.shape[1]] = training_batch.noisy_model_input + v_pred * delta
        hidden_minus[:, : v_pred.shape[1]] = training_batch.noisy_model_input - v_pred * delta

        t_plus = (training_batch.timesteps + epsilon).clamp(
            max=float(num_train_timesteps)
        )
        t_minus = (training_batch.timesteps - epsilon).clamp(min=0.0)
        pred_plus = self._flowmap_forward(
            training_batch, hidden_plus, t_plus, training_batch.r_timesteps
        )
        pred_minus = self._flowmap_forward(
            training_batch, hidden_minus, t_minus, training_batch.r_timesteps
        )
        return (pred_plus - pred_minus) / (2 * epsilon)

    def _transformer_forward_and_compute_loss(self, training_batch):
        if vsa_available and os.environ.get("TRAINER_ATTENTION_BACKEND") == "VIDEO_SPARSE_ATTN":
            assert training_batch.attn_metadata is not None
        else:
            assert training_batch.attn_metadata is None

        with set_forward_context(
            current_timestep=training_batch.current_timestep,
            attn_metadata=training_batch.attn_metadata,
        ):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                model_pred = self.transformer(**training_batch.input_kwargs)[0]

            v_pred = training_batch.noise - training_batch.latents
            dF_dt = self._compute_central_difference(training_batch, v_pred)
            t_minus_r = (training_batch.timesteps - training_batch.r_timesteps).to(
                device=model_pred.device, dtype=model_pred.dtype
            ) / self.noise_scheduler.config.num_train_timesteps
            t_minus_r = rearrange(
                t_minus_r, "(B D) -> B 1 D 1 1", D=training_batch.latents.shape[2]
            )
            target = v_pred - t_minus_r * dF_dt

            i2v_mask = training_batch.i2v_mask
            if training_batch.select_window_out_flag == 1 and self.causal:
                i2v_mask[:, :, :-4, ...] = 0
            assert model_pred.shape == target.shape, (
                f"model_pred.shape: {model_pred.shape}, target.shape: {target.shape}"
            )
            diff = (model_pred.float() * i2v_mask - target.float() * i2v_mask) ** 2
            self._record_flowmap_debug_metrics(
                training_batch,
                model_pred=model_pred,
                target=target,
                v_pred=v_pred,
                dF_dt=dF_dt,
                t_minus_r=t_minus_r,
                i2v_mask=i2v_mask,
                diff=diff,
            )
            loss = (
                diff.sum()
                / max(i2v_mask.sum(), 1)
                / self.training_args.gradient_accumulation_steps
            )
            loss.backward()
            avg_loss = loss.detach().clone()

        dist.all_reduce(avg_loss, op=dist.ReduceOp.MAX)
        training_batch.total_loss += avg_loss.item()
        return training_batch


def main(args) -> None:
    logger.info("Starting flow-map causal student training pipeline...")
    pipeline = FlowMapHunyuanTrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args
    )
    pipeline.train()
    logger.info("Flow-map causal student training pipeline done")


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = TrainerArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.dit_cpu_offload = False
    main(args)
