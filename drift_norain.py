"""Measure horizontal scene drift with RAIN REMOVED.
Rain = thin (1-2px) fast diagonal streaks -> heavy downsample averages it away,
while the big blocky Minecraft terrain survives. We measure on the downsampled
frames so the rain can't bias the result. HUD cropped.
Reports drift at full-res vs downsampled so the rain artifact (if any) is visible.
dx < 0 => LEFT, dx > 0 => RIGHT.
"""
import sys
import cv2
import numpy as np

vids = {
    "gt_decoded(latent)": "outputs/train_infer_orig/seed_186_part_186_0_129/gt_decoded.mp4",
    "gt_sample0(raw)":    "outputs/dmd_init_T32_slicelast4_fixed/val/gt_sample0.mp4",
    "orig_4step":         "outputs/train_infer_orig/seed_186_part_186_0_129/gen.mp4",
}


def load_gray(path):
    cap = cv2.VideoCapture(path)
    fs = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fs.append(f)
    cap.release()
    h, w = fs[0].shape[:2]
    hud = int(h * 0.86)
    return [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[:hud, :] for f in fs]


def accum_phase(grays):
    acc = 0.0
    win = cv2.createHanningWindow((grays[0].shape[1], grays[0].shape[0]), cv2.CV_32F)
    for a, b in zip(grays[:-1], grays[1:]):
        (sx, sy), _ = cv2.phaseCorrelate(np.float32(a), np.float32(b), win)
        acc += sx
    return acc


for name, path in vids.items():
    g = load_gray(path)
    n = len(g)
    h, w = g[0].shape
    # downsample factor 8 -> rain (~1-2px) gone; scale dx back to full-res px
    ds = [cv2.resize(x, (w // 8, h // 8), interpolation=cv2.INTER_AREA) for x in g]
    full = accum_phase(g)
    low = accum_phase(ds) * 8.0
    # also direct f0 vs f_last on downsampled (single robust global shift)
    win = cv2.createHanningWindow((ds[0].shape[1], ds[0].shape[0]), cv2.CV_32F)
    (sx0, sy0), resp = cv2.phaseCorrelate(np.float32(ds[0]), np.float32(ds[-1]), win)
    print(f"{name:22s} n={n} {w}x{h}")
    print(f"   full-res  accumulated dx = {full:+8.1f}px  ({'LEFT' if full<0 else 'RIGHT'})")
    print(f"   downsampled (no rain)   = {low:+8.1f}px  ({'LEFT' if low<0 else 'RIGHT'})")
    print(f"   f0->f{n-1} direct (ds)   dx = {sx0*8:+8.1f}px  resp={resp:.3f}  ({'LEFT' if sx0<0 else 'RIGHT'})")
