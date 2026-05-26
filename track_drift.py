"""Rigorous horizontal-drift check for ONE video.
Two independent methods, HUD (bottom hotbar) cropped out:
  (1) Lucas-Kanade feature tracking from frame 0 -> report net dx of features
      that survive to the last frame (median, robust).
  (2) consecutive-frame phase correlation, accumulated.
Sign: dx < 0  => content moves LEFT ; dx > 0 => content moves RIGHT.
Also dumps a strip of frames with the surviving tracks drawn.
"""
import sys
import cv2
import numpy as np

path = sys.argv[1]
out_strip = sys.argv[2] if len(sys.argv) > 2 else None

cap = cv2.VideoCapture(path)
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
cap.release()
n = len(frames)
h, w = frames[0].shape[:2]
hud = int(h * 0.86)          # drop bottom 14% (Minecraft hotbar / hearts HUD)
gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[:hud, :] for f in frames]
print(f"{path}\n  {n} frames, {w}x{h}, tracking region y in [0,{hud}]")

# ---- (1) LK feature tracking from frame 0 ----
p0 = cv2.goodFeaturesToTrack(gray[0], maxCorners=400, qualityLevel=0.01,
                             minDistance=8, blockSize=7)
start = p0.copy()
pts = p0.copy()
alive = np.ones(len(p0), dtype=bool)
lk = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
for i in range(1, n):
    nxt, st, err = cv2.calcOpticalFlowPyrLK(gray[i - 1], gray[i], pts, None, **lk)
    st = st.reshape(-1).astype(bool)
    alive &= st
    pts = nxt
dx = (pts[:, 0, 0] - start[:, 0, 0])[alive]
dy = (pts[:, 0, 1] - start[:, 0, 1])[alive]
print(f"  [LK]  {alive.sum()}/{len(p0)} features survived f0->f{n-1}")
print(f"        net dx  median={np.median(dx):+7.1f}px  mean={dx.mean():+7.1f}px"
      f"  ({'LEFT' if np.median(dx) < 0 else 'RIGHT'})")
print(f"        net dy  median={np.median(dy):+7.1f}px (>0 = content moved DOWN)")

# ---- (2) accumulated phase correlation ----
acc = 0.0
win = cv2.createHanningWindow((gray[0].shape[1], gray[0].shape[0]), cv2.CV_32F)
for i in range(1, n):
    a = np.float32(gray[i - 1]); b = np.float32(gray[i])
    (shx, shy), resp = cv2.phaseCorrelate(a, b, win)
    acc += shx
print(f"  [phase] accumulated dx = {acc:+7.1f}px  ({'LEFT' if acc < 0 else 'RIGHT'})")

# ---- strip with tracks ----
if out_strip:
    idxs = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    tiles = []
    for j in idxs:
        # re-run LK up to frame j to get positions there
        pp = start.copy(); al = np.ones(len(start), bool)
        for i in range(1, j + 1):
            nx, st, _ = cv2.calcOpticalFlowPyrLK(gray[i - 1], gray[i], pp, None, **lk)
            al &= st.reshape(-1).astype(bool); pp = nx
        img = frames[j].copy()
        for k in range(len(pp)):
            if al[k] and alive[k]:
                x0, y0 = start[k, 0]; x1, y1 = pp[k, 0]
                cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 1)
                cv2.circle(img, (int(x1), int(y1)), 2, (0, 0, 255), -1)
        cv2.putText(img, f"f{j}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        tiles.append(img)
    cv2.imwrite(out_strip, np.hstack(tiles))
    print(f"  strip -> {out_strip}")
