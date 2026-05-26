"""把 _validate 实际喂给模型的 viewmats/action dump 出来,跟原始 _pose.json
对照。复现 _make_val_batch -> _assemble_dmd_batch 的完整路径(含 bf16 cast)。"""
import json
import os
import numpy as np
import torch

os.environ.setdefault("MODEL_ROOT", "./local_models")
from trainer.training.dmd.dmd_dataset import DMDDataset, _camera_center_normalization

JSON = "datasets/preprocessed_gamefactory_f129/dataset_index.json"
ds = DMDDataset(json_path=JSON, num_training_frames=32)
sample = ds[0]                       # val_sample_idx = 0

w2c = sample["w2c"]                  # [32,4,4]  DMDDataset 输出
action = sample["action"]            # [32]

# --- _assemble_dmd_batch 会做的 bf16 往返 ---
w2c_bf16 = w2c.to(torch.bfloat16).float()
action_bf16 = action.to(torch.bfloat16)

def centers(w2c_arr):
    """每帧相机中心 c = -R^T t (世界系)"""
    c = []
    for m in np.asarray(w2c_arr, dtype=np.float64):
        R, t = m[:3, :3], m[:3, 3]
        c.append(-R.T @ t)
    return np.array(c)

c_ds = centers(w2c.numpy())
print("=== _validate 实际喂给模型的 viewmats (DMDDataset, 已 _camera_center_normalization) ===")
print("帧数:", w2c.shape, " dtype:", w2c.dtype)
print("相机中心 frame0 :", np.round(c_ds[0], 3))
print("相机中心 frame15:", np.round(c_ds[15], 3))
print("相机中心 frame31:", np.round(c_ds[31], 3))
print("总位移 c[31]-c[0]:", np.round(c_ds[31] - c_ds[0], 3))
print("逐帧 x 位移(>0=世界系+x):", np.round(np.diff(c_ds[:, 0]), 3))
print()
print("bf16 往返后 viewmats 与原值最大差:", float((w2c_bf16 - w2c.float()).abs().max()))
print()
print("=== action (DMDDataset _read_action_for_pe) ===")
print("dtype:", action.dtype, " 值:", action.tolist())
print("bf16 cast 后:", action_bf16.tolist(), " dtype:", action_bf16.dtype)
print()

# --- 原始 _pose.json:不归一化,直接看真实轨迹 ---
pose_json = json.load(open(
    "datasets/preprocessed_gamefactory_f129/seed_186_part_186_0_129/seed_186_part_186_0_129_pose.json"))
pose_keys = list(pose_json.keys())
raw_w2c = []
picked = []
for i in range(32):
    k = pose_keys[0] if i == 0 else pose_keys[4 * (i - 1) + 4]
    picked.append(k)
    raw_w2c.append(np.array(pose_json[k]["w2c"]))
raw_w2c = np.array(raw_w2c)
c_raw = centers(raw_w2c)
print("=== 原始 _pose.json 选中的帧 (4*(i-1)+4 映射) ===")
print("选中的 video-frame key:", picked)
print("原始相机中心 frame0 :", np.round(c_raw[0], 3))
print("原始相机中心 frame31:", np.round(c_raw[31], 3))
print("原始总位移 c[31]-c[0]:", np.round(c_raw[31] - c_raw[0], 3))
print("原始逐帧 x 位移:", np.round(np.diff(c_raw[:, 0]), 3))
print()

# --- 归一化后,DMDDataset 输出 vs 手动归一化原始 pose,应当完全一致 ---
raw_norm = _camera_center_normalization(raw_w2c)
diff = np.abs(raw_norm - w2c.numpy()).max()
print("手动归一化原始 pose  vs  DMDDataset 输出  最大差:", diff,
      " -> ", "一致" if diff < 1e-6 else "**不一致!!**")

# 动作方向对照
act_json = json.load(open(
    "datasets/preprocessed_gamefactory_f129/seed_186_part_186_0_129/seed_186_part_186_0_129_action.json"))
ak = list(act_json.keys())
moves = [act_json[ak[0] if i == 0 else ak[4 * (i - 1) + 4]]["move_action"] for i in range(32)]
print("原始 action.json move_action(每个 latent):", moves)
