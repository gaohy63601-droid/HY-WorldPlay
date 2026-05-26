"""用 seed_186_part_186_0_129 的【真实训练 pose】重推,不用 w-31。
走和 DMD validation 完全一样的 _ar_rollout_inner 路径。

  mode=orig50  : 原始 ar_model, 50 步
  mode=orig4   : 原始 ar_model, 4 步 (跟 DMD validation 同步数, 同模型 = step 0 student)
  mode=dmd300  : DMD checkpoint-300 的 generator, 4 步

conditioning(pose/action/image_cond/prompt/vision)全部取自 DMDDataset[0],
即和训练/validation 逐字节一致。
"""
import os
import sys

mode = sys.argv[1]                       # orig50 | orig4 | dmd300 | bi50
assert mode in ("orig50", "orig4", "dmd300", "bi50")
os.environ.setdefault("MODEL_ROOT", "./local_models")

import numpy as np
import torch
import einops
import imageio

from trainer.training.dmd.dmd_dataset import DMDDataset
from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.pipelines.pipeline_utils import retrieve_timesteps
from hyvideo.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from hyvideo.utils.retrieval_context import generate_points_in_sphere

# create_pipeline 依赖全局 infer_state(generate.py 里由 initialize_infer_state 建)
from types import SimpleNamespace
from hyvideo.commons.infer_state import initialize_infer_state
initialize_infer_state(SimpleNamespace(
    sage_blocks_range="0-0", use_sageattn=False, include_patterns="double_blocks",
    enable_torch_compile=False, use_fp8_gemm=False, quant_type="fp8-per-block",
    use_vae_parallel=False,
))

MODEL_PATH = "/home/Yirenteam/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038"
AR_CKPT = "/home/Yirenteam/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/f4c29235647707b571479a69b569e4166f9f5bf8/ar_model/diffusion_pytorch_model.safetensors"
BI_CKPT = "/home/Yirenteam/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/f4c29235647707b571479a69b569e4166f9f5bf8/bidirectional_model/diffusion_pytorch_model.safetensors"
DMD_CKPT = os.environ.get(
    "DMD_CKPT",
    "/raid/yiren/ghy/new/HY-WorldPlay/outputs/dmd_init_T32_slicelast4_fixed/checkpoint-300",
)
JSON = os.environ.get(
    "DATASET_JSON",
    "datasets/preprocessed_gamefactory_f129/dataset_index.json",
)
OUT_TAG = os.environ.get("OUT_TAG", mode)
OUT = f"outputs/realpose_infer/{OUT_TAG}.mp4"
os.makedirs("outputs/realpose_infer", exist_ok=True)
N_STEPS = 50 if mode in ("orig50", "bi50") else 4
device = "cuda"
dtype = torch.bfloat16

# ---------- 1. conditioning: DMDDataset[0] (真实 pose) ----------
ds = DMDDataset(json_path=JSON, num_training_frames=32)
s = ds[0]
T = s["w2c"].shape[0]
C = s["image_cond"].shape[0]
print(f"[{mode}] sample0 T={T} C={C}")

def bd(x):  # add batch dim
    return x.unsqueeze(0)

cond_latents = bd(s["cond_latents"]).to(device, dtype)          # [1,C+1,T,H,W]
viewmats = bd(s["w2c"]).float().to(device)                      # [1,T,4,4]  真实 pose
Ks = bd(s["intrinsic"]).float().to(device)                      # [1,T,3,3]
action = bd(s["action"]).long().to(device)                      # [1,T]  真实动作
prompt_embeds = bd(s["prompt_embed"]).to(device, dtype)
prompt_mask = bd(s["prompt_mask"]).to(device)
vision_states = bd(s["vision_states"]).to(device, dtype)
byt5_states = bd(s["byt5_text_states"]).to(device, dtype)
byt5_mask = bd(s["byt5_text_mask"]).to(device)

cen = -viewmats[0, :, :3, :3].transpose(-1, -2).double().cpu() @ viewmats[0, :, :3, 3].double().cpu().unsqueeze(-1)
print(f"[{mode}] viewmats 相机中心 f0={cen[0,:,0].numpy().round(3)}  f{T-1}={cen[-1,:,0].numpy().round(3)}")

g = torch.Generator(device=device).manual_seed(int(os.environ.get("INFER_SEED", "1")))
_, Cp1, Tn, H, W = cond_latents.shape
initial_noise = torch.randn(1, Cp1 - 1, Tn, H, W, device=device, dtype=dtype, generator=g)

# ---------- 2. pipeline ----------
action_ckpt_for_load = BI_CKPT if mode == "bi50" else AR_CKPT
pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
    pretrained_model_name_or_path=MODEL_PATH,
    transformer_version="480p_i2v",
    create_sr_pipeline=False,
    transformer_dtype=dtype,
    enable_offloading=False,
    action_ckpt=action_ckpt_for_load,
)

if mode == "dmd300":
    print(f"[{mode}] loading DMD generator from {DMD_CKPT}")
    if DMD_CKPT.endswith(".safetensors"):
        # Single-file safetensors (extracted via dcp_gen_to_safetensors.py)
        from safetensors.torch import load_file
        gen_sd = load_file(DMD_CKPT)
        miss, unexp = pipe.transformer.load_state_dict(gen_sd, strict=False)
    else:
        # Full DCP checkpoint dir (training state with optimizer/RNG/etc)
        import torch.distributed.checkpoint as dcp
        gen_sd = pipe.transformer.state_dict()
        dcp.load({"generator": gen_sd}, checkpoint_id=DMD_CKPT)
        miss, unexp = pipe.transformer.load_state_dict(gen_sd, strict=False)
    print(f"[{mode}] loaded DMD generator: missing={len(miss)} unexpected={len(unexp)}")

pipe.transformer = pipe.transformer.to(device).eval()
pipe.execution_device = torch.device(device)
pipe.enable_offloading = False
pipe.target_dtype = dtype
pipe.autocast_enabled = True
pipe._progress_bar_config = {}
pipe._guidance_scale = 1.0

scheduler = FlowMatchDiscreteScheduler(shift=5.0, reverse=True, solver="euler")
timesteps, n_is = retrieve_timesteps(scheduler, N_STEPS, device)
pipe.scheduler = scheduler
pipe.num_inference_steps = n_is
pipe.num_warmup_steps = len(timesteps) - n_is * scheduler.order
pipe.chunk_latent_frames = 4
pipe.chunk_num = Tn // 4
pipe.points_local = generate_points_in_sphere(50000, 8.0).to(device)
print(f"[{mode}] steps={N_STEPS} timesteps={[round(float(t),1) for t in timesteps]} chunks={pipe.chunk_num}")

# ---------- 3. rollout (真实推理路径) ----------
rollout_fn = (
    HunyuanVideo_1_5_Pipeline.bi_rollout
    if mode == "bi50"
    else HunyuanVideo_1_5_Pipeline._ar_rollout_inner
)
with torch.no_grad():
    clean = rollout_fn(
        pipe,
        latents=initial_noise.clone(),
        timesteps=timesteps,
        prompt_embeds=prompt_embeds,
        prompt_mask=prompt_mask,
        vision_states=vision_states,
        cond_latents=cond_latents,
        task_type="i2v",
        extra_kwargs={"byt5_text_states": byt5_states, "byt5_text_mask": byt5_mask},
        viewmats=viewmats,
        Ks=Ks,
        action=action,
        device=device,
    )
print(f"[{mode}] rollout done, latent {tuple(clean.shape)} abs_mean={float(clean.abs().mean()):.4f}")

# ---------- 4. decode ----------
vae = pipe.vae.to(device).float().eval()
with torch.no_grad():
    z = clean.to(device, torch.float32)
    if getattr(vae.config, "shift_factor", None):
        z = z / vae.config.scaling_factor + vae.config.shift_factor
    else:
        z = z / vae.config.scaling_factor
    frames = vae.decode(z, return_dict=False)[0]
    frames = (frames / 2 + 0.5).clamp(0, 1).float().cpu()
    if frames.ndim == 5:
        frames = frames[0]
vid = (frames * 255).clamp(0, 255).to(torch.uint8)
vid = einops.rearrange(vid, "c f h w -> f h w c").numpy()
imageio.mimwrite(OUT, vid, fps=16)
print(f"[{mode}] wrote {OUT}  frames={vid.shape}")
