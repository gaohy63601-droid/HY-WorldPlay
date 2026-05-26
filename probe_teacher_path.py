"""Probe: DMD teacher full-32 forward (A) vs bi_rollout-style chunked forward (B).

Same noisy xt, pose, action.

Path A: single forward on 32-latent xt, uniform timestep.
Path B: for each chunk of 4 latent
            mem_idx = select_aligned_memory_frames(...)
            input = cat([clean[mem_idx], xt[cur:cur+4]], dim=T)
            timestep = [stabilization-1]*n_mem + [t]*4
            BI forward
            v_B[cur:cur+4] = output[n_mem:]
"""
import os, sys, importlib.util
import numpy as np
import torch
import torch.nn.functional as F

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
from hyvideo.utils.retrieval_context import (
    generate_points_in_sphere, select_aligned_memory_frames,
)

spec = importlib.util.spec_from_file_location("dmd_dataset", "trainer/training/dmd/dmd_dataset.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ds = mod.DMDDataset(
    json_path="datasets/preprocessed_gamefactory_f129_fixed/dataset_index.json",
    num_training_frames=32,
)
s = ds[0]
print(f"[probe] sample={os.path.basename(ds.json_data[0]['latent_path'])}")

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
points_local = generate_points_in_sphere(50000, 8.0).to(device)

def bd(x): return x.unsqueeze(0)
cond_latents  = bd(s["cond_latents"]).to(device, dtype)
viewmats      = bd(s["w2c"]).float().to(device)
Ks            = bd(s["intrinsic"]).float().to(device)
action        = bd(s["action"]).long().to(device)
prompt_embeds = bd(s["prompt_embed"]).to(device, dtype)
prompt_mask   = bd(s["prompt_mask"]).to(device)
vision_states = bd(s["vision_states"]).to(device, dtype)
byt5_states   = bd(s["byt5_text_states"]).to(device, dtype)
byt5_mask     = bd(s["byt5_text_mask"]).to(device)
extra_kwargs  = {"byt5_text_states": byt5_states, "byt5_text_mask": byt5_mask}

latent_pt = torch.load(ds.json_data[0]["latent_path"], map_location="cpu", weights_only=True)
clean_full = latent_pt["latent"][0, :, :32].to(device, dtype)
C, T, H, W = clean_full.shape
print(f"[probe] clean_full shape={tuple(clean_full.shape)}")

g = torch.Generator(device=device).manual_seed(42)
eps = torch.randn(1, C, T, H, W, device=device, dtype=dtype, generator=g)

NUM_TRAIN_T = 1000
t = 500
sigma = t / NUM_TRAIN_T
xt = (1.0 - sigma) * clean_full[None] + sigma * eps
print(f"[probe] t={t}  sigma={sigma:.3f}  ||xt||={float(xt.norm()):.2f}")

def fwd(latents_5d, ts_per_frame, viewmats_b, Ks_b, action_b, cond_latents_b):
    latents_concat = torch.cat([latents_5d, cond_latents_b], dim=1)
    t_expand_txt = torch.tensor([t], device=device, dtype=ts_per_frame.dtype)
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
        out = real_score(
            bi_inference=True,
            ar_txt_inference=False,
            ar_vision_inference=False,
            hidden_states=latents_concat,
            timestep=ts_per_frame.to(device),
            timestep_txt=t_expand_txt,
            text_states=prompt_embeds,
            text_states_2=None,
            encoder_attention_mask=prompt_mask,
            timestep_r=None,
            vision_states=vision_states,
            mask_type="i2v",
            guidance=None,
            return_dict=False,
            extra_kwargs=extra_kwargs,
            viewmats=viewmats_b.float(),   # KEEP float32 — internal linalg.inv doesn't support bf16
            Ks=Ks_b.float(),
            action=action_b.to(dtype).reshape(-1),
        )
    return out[0]

print("\n[A] full 32, uniform t ...")
with torch.no_grad():
    ts_A = torch.full((T,), t, device=device, dtype=torch.float32)
    v_A = fwd(xt, ts_A, viewmats, Ks, action, cond_latents[:, :, :T])
print(f"[A] v_A shape={tuple(v_A.shape)}  ||v_A||={float(v_A.norm()):.2f}")

print("\n[B] chunked (mem=clean + cur=xt, mixed t) ...")
CHUNK = 4
N_CHUNK = T // CHUNK
T_CTX = 14
v_B = torch.zeros_like(v_A)

with torch.no_grad():
    for ci in range(N_CHUNK):
        cs, ce = ci * CHUNK, (ci + 1) * CHUNK
        if ci == 0:
            latent_in = xt[:, :, cs:ce]
            cond_in   = cond_latents[:, :, cs:ce]
            ts_in     = torch.full((CHUNK,), t, device=device, dtype=torch.float32)
            vm_in     = viewmats[:, cs:ce]
            Ks_in     = Ks[:, cs:ce]
            act_in    = action[:, cs:ce]
            v_chunk   = fwd(latent_in, ts_in, vm_in, Ks_in, act_in, cond_in)
            v_B[:, :, cs:ce] = v_chunk
            n_mem = 0
        else:
            mem_idx = select_aligned_memory_frames(
                viewmats[0].cpu().numpy(),
                cs, memory_frames=20, temporal_context_size=12,
                pred_latent_size=4, points_local=points_local, device=device,
            )
            mem_idx = sorted([i for i in set(mem_idx) if 0 <= i < cs])
            n_mem = len(mem_idx)
            latent_in = torch.cat([clean_full[None][:, :, mem_idx], xt[:, :, cs:ce]], dim=2)
            cond_in   = cond_latents[:, :, :n_mem + CHUNK]
            ts_in     = torch.cat([
                torch.full((n_mem,), T_CTX, device=device, dtype=torch.float32),
                torch.full((CHUNK,),  t,     device=device, dtype=torch.float32),
            ])
            vm_in     = torch.cat([viewmats[:, mem_idx], viewmats[:, cs:ce]], dim=1)
            Ks_in     = torch.cat([Ks[:, mem_idx], Ks[:, cs:ce]], dim=1)
            act_in    = torch.cat([action[:, mem_idx], action[:, cs:ce]], dim=1)
            v_chunk   = fwd(latent_in, ts_in, vm_in, Ks_in, act_in, cond_in)
            v_B[:, :, cs:ce] = v_chunk[:, :, n_mem:]
        print(f"  chunk {ci}: n_mem={n_mem}  ||v_chunk[cur]||={float(v_chunk[:, :, -CHUNK:].norm()):.2f}")

print(f"\n[B] v_B shape={tuple(v_B.shape)}  ||v_B||={float(v_B.norm()):.2f}")

print("\n" + "=" * 72)
print("COMPARISON: PATH A (DMD teacher full-32) vs PATH B (chunked bi_rollout-style)")
print("=" * 72)
vA_f = v_A.float()
vB_f = v_B.float()
diff = vA_f - vB_f
print(f"global ||v_A||={vA_f.norm():.2f}  ||v_B||={vB_f.norm():.2f}  ||diff||={diff.norm():.2f}  rel={float(diff.norm()/vA_f.norm()):.4f}")
print(f"global cos(v_A, v_B) = {F.cosine_similarity(vA_f.flatten()[None], vB_f.flatten()[None]).item():.4f}")

print(f"\n{'frame':>5} {'||v_A||':>9} {'||v_B||':>9} {'||diff||':>9} {'rel':>7} {'cos':>7}")
for fi in range(T):
    a = vA_f[0, :, fi]
    b = vB_f[0, :, fi]
    cos = F.cosine_similarity(a.flatten()[None], b.flatten()[None]).item()
    rel = float((a - b).norm() / a.norm().clamp_min(1e-9))
    print(f"{fi:>5} {float(a.norm()):>9.2f} {float(b.norm()):>9.2f} {float((a-b).norm()):>9.2f} {rel:>7.4f} {cos:>7.4f}")

x0_A = xt.float() - sigma * vA_f
x0_B = xt.float() - sigma * vB_f
print(f"\nx0 reconstruction (vs GT clean latent):")
print(f"  ||x0_A - clean|| / ||clean|| = {float((x0_A - clean_full[None].float()).norm() / clean_full.float().norm()):.4f}")
print(f"  ||x0_B - clean|| / ||clean|| = {float((x0_B - clean_full[None].float()).norm() / clean_full.float().norm()):.4f}")
print(f"  ||x0_A - x0_B||  / ||x0_A||  = {float((x0_A - x0_B).norm() / x0_A.norm()):.4f}")
print(f"  cos(x0_A, x0_B) = {F.cosine_similarity(x0_A.flatten()[None], x0_B.flatten()[None]).item():.4f}")
print("\n[probe] done.")
