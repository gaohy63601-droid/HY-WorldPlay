"""Decode the *canonical* GT for a preprocessed segment: the VAE latent stored
in datasets/preprocessed_gamefactory_f129/<seg>/<seg>_latent.pt is exactly what
DMD training feeds as the real-data target. Decoding it removes all ambiguity
about "which raw-video frames are the GT". Decode logic byte-mirrors
dmd_init_training_pipeline._decode_latent_to_mp4.
"""
import sys
import torch
import einops
import imageio

seg = sys.argv[1] if len(sys.argv) > 1 else "seed_186_part_186_0_129"
latent_path = f"datasets/preprocessed_gamefactory_f129/{seg}/{seg}_latent.pt"
out_path = sys.argv[2] if len(sys.argv) > 2 else f"outputs/train_infer_orig/{seg}/gt_decoded.mp4"
vae_dir = "/home/Yirenteam/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038/vae"

device = "cuda"
from hyvideo.models.autoencoders.hunyuanvideo_15_vae_w_cache import AutoencoderKLConv3D

vae = AutoencoderKLConv3D.from_pretrained(vae_dir, torch_dtype=torch.float32).to(device).eval()
for p in vae.parameters():
    p.requires_grad_(False)

d = torch.load(latent_path, map_location="cpu", weights_only=False)
latent = d["latent"]
print("latent", latent.shape, latent.dtype, "scaling_factor", vae.config.scaling_factor,
      "shift_factor", getattr(vae.config, "shift_factor", None))

with torch.no_grad():
    z = latent.to(device, dtype=torch.float32)
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

imageio.mimwrite(out_path, vid, fps=16)
print("wrote", out_path, "frames", vid.shape)
