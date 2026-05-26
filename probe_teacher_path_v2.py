"""Probe v2: full simulation of DMD training teacher call.

Step 1: AR 4-step generator rollout → clean_estimate (matches DMD's
        self.pipe.rollout output)
Step 2: For multiple sampled timesteps (covering DMD _re_noise's range):
        - noisy = (1-σ)·clean_estimate + σ·ε   (= DMD _re_noise)
        - Path A: full 32 forward of BI (= current DMD teacher)
        - Path B: chunked, memory = clean_estimate (= Tencent bi_rollout
                  but using generator's clean instead of GT)
        - Compare cos, rel_diff, x0 reconstruction

Skip CFG for simplicity — applies symmetrically to both paths so doesn't
affect A-vs-B comparison.
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
from hyvideo.pipelines.pipeline_utils import retrieve_timesteps
from hyvideo.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from hyvideo.utils.retrieval_context import generate_points_in_sphere, select_aligned_memory_frames

device = "cuda"
dtype  = torch.bfloat16

MODEL_PATH = "/home/Yirenteam/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038"
AR_CKPT    = "/home/Yirenteam/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/f4c29235647707b571479a69b569e4166f9f5bf8/ar_model/diffusion_pytorch_model.safetensors"
BI_CKPT    = "/home/Yirenteam/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/f4c29235647707b571479a69b569e4166f9f5bf8/bidirectional_model/diffusion_pytorch_model.safetensors"

# ---- DMDDataset[0] fixed ----
spec = importlib.util.spec_from_file_location("dmd_dataset", "trainer/training/dmd/dmd_dataset.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ds = mod.DMDDataset(json_path="datasets/preprocessed_gamefactory_f129_fixed/dataset_index.json", num_training_frames=32)
s = ds[0]
print(f"[probe2] sample={os.path.basename(ds.json_data[0]['latent_path'])}")

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

_, Cp1, Tn, H, W = cond_latents.shape
C = Cp1 - 1
points_local = generate_points_in_sphere(50000, 8.0).to(device)

# ====================== STEP 1: AR 4-step rollout → clean_estimate ======================
print("\n[probe2] loading AR generator (4-step student) ...")
pipe_ar = HunyuanVideo_1_5_Pipeline.create_pipeline(
    pretrained_model_name_or_path=MODEL_PATH,
    transformer_version="480p_i2v",
    create_sr_pipeline=False,
    transformer_dtype=dtype,
    enable_offloading=False,
    action_ckpt=AR_CKPT,
)
pipe_ar.transformer = pipe_ar.transformer.to(device).eval()
pipe_ar.execution_device = torch.device(device)
pipe_ar.enable_offloading = False
pipe_ar.target_dtype = dtype
pipe_ar.autocast_enabled = True
pipe_ar._progress_bar_config = {}
pipe_ar._guidance_scale = 1.0
pipe_ar.points_local = points_local

scheduler_ar = FlowMatchDiscreteScheduler(shift=5.0, reverse=True, solver="euler")
ts_ar, _ = retrieve_timesteps(scheduler_ar, 4, device)
pipe_ar.scheduler = scheduler_ar
pipe_ar.num_inference_steps = 4
pipe_ar.num_warmup_steps = 0
pipe_ar.chunk_latent_frames = 4
pipe_ar.chunk_num = Tn // 4

g = torch.Generator(device=device).manual_seed(1)
initial_noise = torch.randn(1, C, Tn, H, W, device=device, dtype=dtype, generator=g)

print(f"[probe2] AR 4-step rollout (32 latent, chunked + memory) ...")
with torch.no_grad():
    clean_estimate = HunyuanVideo_1_5_Pipeline._ar_rollout_inner(
        pipe_ar,
        latents=initial_noise.clone(),
        timesteps=ts_ar,
        prompt_embeds=prompt_embeds,
        prompt_mask=prompt_mask,
        vision_states=vision_states,
        cond_latents=cond_latents,
        task_type="i2v",
        extra_kwargs=extra_kwargs,
        viewmats=viewmats,
        Ks=Ks,
        action=action,
        device=device,
    )
print(f"[probe2] clean_estimate shape={tuple(clean_estimate.shape)}  abs_mean={float(clean_estimate.abs().mean()):.4f}")

# ---- also load GT clean for diagnostic comparison ----
latent_pt = torch.load(ds.json_data[0]["latent_path"], map_location="cpu", weights_only=True)
clean_gt = latent_pt["latent"][0, :, :32].to(device, dtype).unsqueeze(0)
gt_vs_gen_rel = float((clean_estimate - clean_gt).norm() / clean_gt.norm())
gt_vs_gen_cos = F.cosine_similarity(clean_estimate.flatten()[None].float(), clean_gt.flatten()[None].float()).item()
print(f"[probe2] generator clean vs GT clean: rel={gt_vs_gen_rel:.4f}  cos={gt_vs_gen_cos:.4f}")

# ---- free AR ----
del pipe_ar
torch.cuda.empty_cache()

# ====================== STEP 2: load BI for real_score ======================
print("\n[probe2] loading BI (real_score teacher) ...")
pipe_bi = HunyuanVideo_1_5_Pipeline.create_pipeline(
    pretrained_model_name_or_path=MODEL_PATH,
    transformer_version="480p_i2v",
    create_sr_pipeline=False,
    transformer_dtype=dtype,
    enable_offloading=False,
    action_ckpt=BI_CKPT,
)
real_score = pipe_bi.transformer.to(device).eval()

# ====================== forward helper ======================
def fwd(lat_5d, ts_per_frame, vm_b, Ks_b, act_b, cond_in, p_embeds, p_mask, byt5_s, byt5_m, vis_s):
    lat_concat = torch.cat([lat_5d, cond_in], dim=1)
    t_txt = torch.tensor([ts_per_frame[0]], device=device, dtype=ts_per_frame.dtype)
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
        out = real_score(
            bi_inference=True, ar_txt_inference=False, ar_vision_inference=False,
            hidden_states=lat_concat,
            timestep=ts_per_frame.to(device),
            timestep_txt=t_txt,
            text_states=p_embeds, text_states_2=None,
            encoder_attention_mask=p_mask,
            timestep_r=None, vision_states=vis_s,
            mask_type="i2v", guidance=None,
            return_dict=False,
            extra_kwargs={"byt5_text_states": byt5_s, "byt5_text_mask": byt5_m},
            viewmats=vm_b.float(), Ks=Ks_b.float(),
            action=act_b.to(dtype).reshape(-1),
        )
    return out[0]

# ====================== STEP 3: probe across multiple timesteps ======================
NUM_TRAIN_T = 1000
T_CTX = 14
CHUNK = 4
N_CHUNK = Tn // CHUNK

# DMD _re_noise produces sampled t in [min_step=20, max_step=980] (with SD3 shift).
# Probe a representative spread.
test_ts = [50, 200, 500, 800, 950]

print("\n" + "="*78)
print("PROBE v2: path A (full 32 single forward) vs path B (chunked with mem=clean_estimate)")
print(f"clean = AR 4-step generator output (NOT GT)")
print(f"timesteps tested = {test_ts}")
print("="*78)
print(f"\n{'t':>5} {'sigma':>6} {'||v_A||':>10} {'||v_B||':>10} {'cos(v)':>8} {'rel(v)':>8}  {'cos(x0)':>8} {'rel(x0)':>8}")

results = []
g_eps = torch.Generator(device=device).manual_seed(42)
for t in test_ts:
    sigma = t / NUM_TRAIN_T
    eps = torch.randn(1, C, Tn, H, W, device=device, dtype=dtype, generator=g_eps)
    xt = (1.0 - sigma) * clean_estimate + sigma * eps

    # ---- Path A: full 32, uniform t ----
    with torch.no_grad():
        ts_A = torch.full((Tn,), t, device=device, dtype=torch.float32)
        v_A = fwd(xt, ts_A, viewmats, Ks, action, cond_latents[:, :, :Tn],
                  prompt_embeds, prompt_mask, byt5_states, byt5_mask, vision_states)

    # ---- Path B: chunked, mem = clean_estimate ----
    v_B = torch.zeros_like(v_A)
    with torch.no_grad():
        for ci in range(N_CHUNK):
            cs, ce = ci * CHUNK, (ci + 1) * CHUNK
            if ci == 0:
                lat_in = xt[:, :, cs:ce]
                cond_in = cond_latents[:, :, cs:ce]
                ts_in = torch.full((CHUNK,), t, device=device, dtype=torch.float32)
                vm_in = viewmats[:, cs:ce]
                Ks_in = Ks[:, cs:ce]
                act_in = action[:, cs:ce]
                v_chunk = fwd(lat_in, ts_in, vm_in, Ks_in, act_in, cond_in,
                              prompt_embeds, prompt_mask, byt5_states, byt5_mask, vision_states)
                v_B[:, :, cs:ce] = v_chunk
            else:
                mem_idx = select_aligned_memory_frames(
                    viewmats[0].cpu().numpy(),
                    cs, memory_frames=20, temporal_context_size=12,
                    pred_latent_size=4, points_local=points_local, device=device,
                )
                mem_idx = sorted([i for i in set(mem_idx) if 0 <= i < cs])
                n_mem = len(mem_idx)
                lat_in = torch.cat([clean_estimate[:, :, mem_idx], xt[:, :, cs:ce]], dim=2)
                cond_in = cond_latents[:, :, :n_mem + CHUNK]
                ts_in = torch.cat([
                    torch.full((n_mem,), T_CTX, device=device, dtype=torch.float32),
                    torch.full((CHUNK,),  t,     device=device, dtype=torch.float32),
                ])
                vm_in = torch.cat([viewmats[:, mem_idx], viewmats[:, cs:ce]], dim=1)
                Ks_in = torch.cat([Ks[:, mem_idx], Ks[:, cs:ce]], dim=1)
                act_in = torch.cat([action[:, mem_idx], action[:, cs:ce]], dim=1)
                v_chunk = fwd(lat_in, ts_in, vm_in, Ks_in, act_in, cond_in,
                              prompt_embeds, prompt_mask, byt5_states, byt5_mask, vision_states)
                v_B[:, :, cs:ce] = v_chunk[:, :, n_mem:]

    # ---- compare ----
    vA_f = v_A.float()
    vB_f = v_B.float()
    cos_v = F.cosine_similarity(vA_f.flatten()[None], vB_f.flatten()[None]).item()
    rel_v = float((vA_f - vB_f).norm() / vA_f.norm())

    x0_A = xt.float() - sigma * vA_f
    x0_B = xt.float() - sigma * vB_f
    cos_x0 = F.cosine_similarity(x0_A.flatten()[None], x0_B.flatten()[None]).item()
    rel_x0 = float((x0_A - x0_B).norm() / x0_A.norm())

    print(f"{t:>5d} {sigma:>6.3f} {float(vA_f.norm()):>10.2f} {float(vB_f.norm()):>10.2f} "
          f"{cos_v:>8.4f} {rel_v:>8.4f}  {cos_x0:>8.4f} {rel_x0:>8.4f}")
    results.append((t, cos_v, rel_v, cos_x0, rel_x0))

print("\n[probe2] done.")
print(f"\nSummary across timesteps:")
print(f"  cos(v):  min={min(r[1] for r in results):.4f}  max={max(r[1] for r in results):.4f}")
print(f"  rel(v):  min={min(r[2] for r in results):.4f}  max={max(r[2] for r in results):.4f}")
print(f"  cos(x0): min={min(r[3] for r in results):.4f}  max={max(r[3] for r in results):.4f}")
print(f"  rel(x0): min={min(r[4] for r in results):.4f}  max={max(r[4] for r in results):.4f}")
