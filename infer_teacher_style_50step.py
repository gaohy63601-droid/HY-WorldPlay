"""Teacher-style BI inference: 50 step denoise, full 32 latent single forward
per step, NO chunking, NO memory selection.

Matches DMD teacher's call pattern (single forward on full T) but iterates 50
steps to get a final video instead of just one velocity prediction.

CFG scale = 6.0 (= run_bi_50step.sh default = DMD real_guidance_scale).
"""
import os, sys, importlib.util
import numpy as np
import torch
import einops
import imageio

os.environ.setdefault("MODEL_ROOT", "./local_models")
sys.path.insert(0, "/raid/yiren/ghy/new/HY-WorldPlay")

from types import SimpleNamespace
from hyvideo.commons.infer_state import initialize_infer_state
initialize_infer_state(SimpleNamespace(
    sage_blocks_range="0-0", use_sageattn=False, include_patterns="double_blocks",
    enable_torch_compile=False, use_fp8_gemm=False, quant_type="fp8-per-block",
    use_vae_parallel=False,
))

from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.pipelines.pipeline_utils import retrieve_timesteps
from hyvideo.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler

# ---- load fixed DMDDataset (seed_186 first) ----
spec = importlib.util.spec_from_file_location("dmd_dataset", "trainer/training/dmd/dmd_dataset.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ds = mod.DMDDataset(
    json_path="datasets/preprocessed_gamefactory_f129_fixed/dataset_index.json",
    num_training_frames=32,
)
s = ds[0]
print(f"[teacher50] sample={os.path.basename(ds.json_data[0]['latent_path'])}")

device = "cuda"
dtype  = torch.bfloat16

MODEL_PATH = "/home/Yirenteam/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038"
BI_CKPT    = "/home/Yirenteam/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/f4c29235647707b571479a69b569e4166f9f5bf8/bidirectional_model/diffusion_pytorch_model.safetensors"

pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
    pretrained_model_name_or_path=MODEL_PATH,
    transformer_version="480p_i2v",
    create_sr_pipeline=False,
    transformer_dtype=dtype,
    enable_offloading=False,
    action_ckpt=BI_CKPT,
)
real_score = pipe.transformer.to(device).eval()

# ---- batch ----
def bd(x): return x.unsqueeze(0)
cond_latents  = bd(s["cond_latents"]).to(device, dtype)         # [1,C+1,T,H,W]  (=33 channels with i2v mask)
viewmats      = bd(s["w2c"]).float().to(device)                 # [1,T,4,4]
Ks            = bd(s["intrinsic"]).float().to(device)
action        = bd(s["action"]).long().to(device)
prompt_embeds = bd(s["prompt_embed"]).to(device, dtype)
prompt_mask   = bd(s["prompt_mask"]).to(device)
vision_states = bd(s["vision_states"]).to(device, dtype)
byt5_states   = bd(s["byt5_text_states"]).to(device, dtype)
byt5_mask     = bd(s["byt5_text_mask"]).to(device)
extra_kwargs  = {"byt5_text_states": byt5_states, "byt5_text_mask": byt5_mask}

# ---- uncond (drop text only, same as DMD teacher CFG) ----
neg = torch.load("/raid/yiren/ghy/new/HY-WorldPlay/local_models/hunyuan_neg_prompt.pt", map_location="cpu", weights_only=True)
neg_byt5 = torch.load("/raid/yiren/ghy/new/HY-WorldPlay/local_models/hunyuan_neg_byt5_prompt.pt", map_location="cpu", weights_only=True)
uncond_prompt_embeds = neg["negative_prompt_embeds"].to(device, dtype)
uncond_prompt_mask   = neg["negative_prompt_mask"].to(device)
uncond_byt5_states   = neg_byt5["byt5_text_states"].to(device, dtype)
uncond_byt5_mask     = neg_byt5["byt5_text_mask"].to(device)
print(f"[teacher50] uncond prompt embed shape={tuple(uncond_prompt_embeds.shape)}")

# ---- initial noise ----
_, Cp1, Tn, H, W = cond_latents.shape
C = Cp1 - 1
g = torch.Generator(device=device).manual_seed(1)
latents = torch.randn(1, C, Tn, H, W, device=device, dtype=dtype, generator=g)
print(f"[teacher50] initial noise shape={tuple(latents.shape)}")

# ---- scheduler ----
N_STEPS = 50
CFG = 6.0
scheduler = FlowMatchDiscreteScheduler(shift=5.0, reverse=True, solver="euler")
timesteps, _ = retrieve_timesteps(scheduler, N_STEPS, device)
print(f"[teacher50] N_STEPS={N_STEPS}  CFG={CFG}  shift=5.0")
print(f"[teacher50] timesteps[:8]={[round(float(t),1) for t in timesteps[:8]]} ... last={float(timesteps[-1]):.1f}")

# ---- forward helper ----
def fwd(lat, ts_per_frame, p_embeds, p_mask, byt5_s, byt5_m):
    """Single transformer forward on full T. lat: [1,C,T,H,W]"""
    lat_with_cond = torch.cat([lat, cond_latents], dim=1)   # [1, C+(C+1), T, H, W]
    t_txt = torch.tensor([ts_per_frame[0]], device=device, dtype=ts_per_frame.dtype)
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
        out = real_score(
            bi_inference=True,
            ar_txt_inference=False,
            ar_vision_inference=False,
            hidden_states=lat_with_cond,
            timestep=ts_per_frame,
            timestep_txt=t_txt,
            text_states=p_embeds,
            text_states_2=None,
            encoder_attention_mask=p_mask,
            timestep_r=None,
            vision_states=vision_states,
            mask_type="i2v",
            guidance=None,
            return_dict=False,
            extra_kwargs={"byt5_text_states": byt5_s, "byt5_text_mask": byt5_m},
            viewmats=viewmats.float(),
            Ks=Ks.float(),
            action=action.to(dtype).reshape(-1),
        )
    return out[0]    # velocity, [1, C, T, H, W]

# ---- iterative 50-step denoise, no chunk no memory ----
print("\n[teacher50] iterating 50 steps ...")
from tqdm import tqdm
with torch.no_grad():
    for i, t in enumerate(tqdm(timesteps)):
        ts = torch.full((Tn,), t, device=device, dtype=timesteps.dtype)
        v_cond   = fwd(latents, ts, prompt_embeds, prompt_mask, byt5_states, byt5_mask)
        v_uncond = fwd(latents, ts, uncond_prompt_embeds, uncond_prompt_mask, uncond_byt5_states, uncond_byt5_mask)
        v = v_uncond + CFG * (v_cond - v_uncond)
        latents = scheduler.step(v, t, latents, return_dict=False)[0]

print(f"\n[teacher50] denoise done. latents shape={tuple(latents.shape)}  abs_mean={float(latents.abs().mean()):.4f}")

# ---- VAE decode ----
print("[teacher50] VAE decoding ...")
vae = pipe.vae.to(device).float().eval()
with torch.no_grad():
    z = latents.to(device, torch.float32)
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

OUT = "outputs/realpose_infer/teacher50.mp4"
os.makedirs("outputs/realpose_infer", exist_ok=True)
imageio.mimwrite(OUT, vid, fps=16)
print(f"[teacher50] wrote {OUT}  frames={vid.shape}")
