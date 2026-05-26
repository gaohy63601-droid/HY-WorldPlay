"""Accumulated horizontal optical-flow drift for a set of videos.
Convention: drift > 0  => scene content slides RIGHT across the frame
            drift < 0  => scene content slides LEFT  across the frame
(Strafe-right action makes the world slide LEFT, i.e. negative drift.)
Uses Farneback dense flow; per frame-pair takes the spatial MEDIAN of the
horizontal component (robust to rain streaks / local motion)."""
import sys
import cv2
import numpy as np

VIDEOS = {
    "GT_canonical (decoded latent)": "outputs/train_infer_orig/seed_186_part_186_0_129/gt_decoded.mp4",
    "GT_old clip (raw frames 0-128)": "outputs/dmd_init_T32_slicelast4_fixed/val/gt_sample0.mp4",
    "orig model @ 4-step":            "outputs/train_infer_orig/seed_186_part_186_0_129/gen.mp4",
    "orig model @ 50-step":           "outputs/orig_50step_sample0/gen.mp4",
    "DMD @ step300 (4-step val)":     "outputs/dmd_init_T32_slicelast4_fixed/val/step_000300.mp4",
}


def read_frames(path):
    cap = cv2.VideoCapture(path)
    fs = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fs.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    return fs


def drift(path):
    fs = read_frames(path)
    if len(fs) < 2:
        return None
    h, w = fs[0].shape
    per = []
    for a, b in zip(fs[:-1], fs[1:]):
        if b.shape != (h, w):
            b = cv2.resize(b, (w, h))
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        per.append(float(np.median(flow[..., 0])))
    per = np.array(per)
    return len(fs), w, per.sum(), per.mean()


for name, path in VIDEOS.items():
    r = drift(path)
    if r is None:
        print(f"{name:34s}  MISSING/empty -> {path}")
        continue
    n, w, total, mean = r
    direction = "LEFT " if total < 0 else "RIGHT"
    print(f"{name:34s}  frames={n:3d} w={w:3d}  accum_dx={total:+8.1f}px ({direction})  per-frame={mean:+.3f}px")
