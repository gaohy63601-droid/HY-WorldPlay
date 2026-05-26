"""X-T slit-scan: take a few horizontal rows, stack each row over all frames.
In the output, x-axis = image column, y-axis = time (top=frame0, bottom=last).
  streaks tilt like  /  -> content moved LEFT
  streaks tilt like  \\  -> content moved RIGHT
  streaks vertical |     -> no horizontal motion
No optical flow, no tracking — just raw pixels. Unambiguous.
"""
import sys
import cv2
import numpy as np

vids = {
    "gt_decoded(latent)": "outputs/train_infer_orig/seed_186_part_186_0_129/gt_decoded.mp4",
    "gt_sample0(raw)":     "outputs/dmd_init_T32_slicelast4_fixed/val/gt_sample0.mp4",
    "orig_4step":          "outputs/train_infer_orig/seed_186_part_186_0_129/gen.mp4",
}
out = "outputs/train_infer_orig/seed_186_part_186_0_129/_xt_slices.png"

panels = []
for name, path in vids.items():
    cap = cv2.VideoCapture(path)
    fs = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fs.append(f)
    cap.release()
    if not fs:
        print("MISSING", path); continue
    h, w = fs[0].shape[:2]
    # normalise width so panels align
    W = 832
    fs = [cv2.resize(f, (W, int(h * W / w))) for f in fs]
    hh = fs[0].shape[0]
    rows = [int(hh * r) for r in (0.30, 0.50, 0.70)]   # 3 rows: upper / mid / lower
    blocks = []
    for ry in rows:
        # each frame contributes a 3px-tall band so the image isn't too thin
        xt = np.stack([f[ry] for f in fs], axis=0)       # [T, W, 3]
        xt = np.repeat(xt, 3, axis=0)
        cv2.putText(xt, f"{name} y={ry}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        blocks.append(xt)
        blocks.append(np.full((6, W, 3), 255, np.uint8))
    panels.append(np.vstack(blocks))
    panels.append(np.full((14, W, 3), 128, np.uint8))
    print(f"{name}: {len(fs)} frames {w}x{h}")

cv2.imwrite(out, np.vstack(panels))
print("wrote", out)
