# SPDX-License-Identifier: Apache-2.0
"""
Lightweight dataset for DMD self-forcing training.

Reads exactly the same preprocessed format produced by
`datasets/hy_preprocess/preprocess_gamefactory_dataset.py`:

    <segment_id>_latent.pt:
        latent           [1, C, T, H, W]    (DISCARDED — DMD doesn't use GT video)
        prompt_embeds    [1, S, D]
        prompt_mask      [1, S]
        image_cond       [1, C, 1, H, W]
        vision_states    [1, V, D_v]
        byt5_text_states [1, B, 1472]
        byt5_text_mask   [1, B]
    <segment_id>_pose.json    (per-frame {intrinsic, w2c})
    <segment_id>_action.json  (per-frame {move_action, view_action})

Behavior == `CameraJsonWMemDataset` mode B only (no memory-frame selection):
returns the first `num_training_frames` *chronological* frames worth of
conditioning, plus a pre-built `cond_latents` of shape [C+1, T, H, W] with
the i2v mask channel appended (frame 0 = image_cond, mask=1; rest = 0).
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset


# Same (trans, rotate) one-hot tuple -> action-id mapping the pre-trained AR
# transformer was trained on. Sourced from
# `trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py`.
_ACTION_MAPPING = {
    (0, 0, 0, 0): 0,
    (1, 0, 0, 0): 1,
    (0, 1, 0, 0): 2,
    (0, 0, 1, 0): 3,
    (0, 0, 0, 1): 4,
    (1, 0, 1, 0): 5,
    (1, 0, 0, 1): 6,
    (0, 1, 1, 0): 7,
    (0, 1, 0, 1): 8,
}


def _camera_center_normalization(w2c: np.ndarray) -> np.ndarray:
    c2w = np.linalg.inv(w2c)
    C0_inv = np.linalg.inv(c2w[0])
    c2w_aligned = np.array([C0_inv @ C for C in c2w])
    return np.linalg.inv(c2w_aligned)


def _one_hot_to_label(one_hot: np.ndarray) -> np.ndarray:
    return np.array([_ACTION_MAPPING[tuple(row.tolist())] for row in one_hot])


class DMDDataset(Dataset):
    """Mode-B-only reader for the DMD init pipeline."""

    def __init__(
        self,
        json_path: str,
        num_training_frames: int = 32,
        cfg_rate: float = 0.0,
        seed: int = 0,
        neg_prompt_path: Optional[str] = None,
        neg_byt5_prompt_path: Optional[str] = None,
    ):
        with open(json_path, "r") as f:
            self.json_data = json.load(f)
        self.num_training_frames = int(num_training_frames)
        self.cfg_rate = float(cfg_rate)
        self.rng = random.Random(seed)

        model_root = os.environ.get("MODEL_ROOT", "./local_models")
        if neg_prompt_path is None:
            neg_prompt_path = os.environ.get(
                "NEG_PROMPT_PT_PATH",
                os.path.join(model_root, "hunyuan_neg_prompt.pt"),
            )
        if neg_byt5_prompt_path is None:
            neg_byt5_prompt_path = os.environ.get(
                "NEG_BYT5_PROMPT_PT_PATH",
                os.path.join(model_root, "hunyuan_neg_byt5_prompt.pt"),
            )

        self.neg_prompt_pt = (
            torch.load(neg_prompt_path, map_location="cpu", weights_only=True)
            if os.path.exists(neg_prompt_path)
            else None
        )
        self.neg_byt5_pt = (
            torch.load(neg_byt5_prompt_path, map_location="cpu", weights_only=True)
            if os.path.exists(neg_byt5_prompt_path)
            else None
        )

    def __len__(self) -> int:
        return len(self.json_data)

    # ------------------------------------------------------ action / camera

    def _read_action_for_pe(
        self, action_path: str, num_latent_frames: int
    ) -> torch.Tensor:
        """Mirror `CameraJsonWMemDataset.__getitem__` 'action_path' branch."""
        action_json = json.load(open(action_path, "r"))
        action_keys = list(action_json.keys())

        trans_one_hot = np.zeros((num_latent_frames, 4), dtype=np.int32)
        rotate_one_hot = np.zeros((num_latent_frames, 4), dtype=np.int32)
        for i in range(1, num_latent_frames):
            t_key = action_keys[4 * (i - 1) + 4]
            move = action_json[t_key]["move_action"]
            view = action_json[t_key]["view_action"]
            if "W" in move and "S" not in move:
                trans_one_hot[i, 0] = 1
            if "S" in move and "W" not in move:
                trans_one_hot[i, 1] = 1
            if "D" in move and "A" not in move:
                trans_one_hot[i, 2] = 1
            if "A" in move and "D" not in move:
                trans_one_hot[i, 3] = 1
            if view == "LR":
                rotate_one_hot[i, 0] = 1
            elif view == "LL":
                rotate_one_hot[i, 1] = 1
            elif view == "LU":
                rotate_one_hot[i, 2] = 1
            elif view == "LD":
                rotate_one_hot[i, 3] = 1

        trans_label = _one_hot_to_label(trans_one_hot)
        rotate_label = _one_hot_to_label(rotate_one_hot)
        return torch.tensor(trans_label * 9 + rotate_label, dtype=torch.long)

    def _action_from_camera(
        self, w2c: np.ndarray, num_latent_frames: int
    ) -> torch.Tensor:
        """Fallback action labels derived from camera deltas (mirrors
        `CameraJsonWMemDataset.__getitem__` 'else' branch)."""
        c2ws = np.linalg.inv(w2c)
        C_inv = np.linalg.inv(c2ws[:-1])
        relative_c2w = np.zeros_like(c2ws)
        relative_c2w[0] = c2ws[0]
        relative_c2w[1:] = C_inv @ c2ws[1:]
        trans_one_hot = np.zeros((num_latent_frames, 4), dtype=np.int32)
        rotate_one_hot = np.zeros((num_latent_frames, 4), dtype=np.int32)
        for i in range(1, num_latent_frames):
            move_dirs = relative_c2w[i, :3, 3]
            move_norm = np.linalg.norm(move_dirs)
            if move_norm > 0.01:
                move_norm_dirs = move_dirs / move_norm
                ang = np.degrees(np.arccos(move_norm_dirs.clip(-1.0, 1.0)))
                if ang[2] < 60:
                    trans_one_hot[i, 0] = 1
                elif ang[2] > 120:
                    trans_one_hot[i, 1] = 1
                if ang[0] < 60:
                    trans_one_hot[i, 2] = 1
                elif ang[0] > 120:
                    trans_one_hot[i, 3] = 1
            rot = R.from_matrix(relative_c2w[i, :3, :3]).as_euler("xyz", degrees=True)
            if rot[1] > 5e-2:
                rotate_one_hot[i, 0] = 1
            elif rot[1] < -5e-2:
                rotate_one_hot[i, 1] = 1
            if rot[0] > 5e-2:
                rotate_one_hot[i, 2] = 1
            elif rot[0] < -5e-2:
                rotate_one_hot[i, 3] = 1

        trans_label = _one_hot_to_label(trans_one_hot)
        rotate_label = _one_hot_to_label(rotate_one_hot)
        return torch.tensor(trans_label * 9 + rotate_label, dtype=torch.long)

    # ------------------------------------------------------------- __getitem__

    def __getitem__(self, idx: int) -> Dict:
        n_attempts = 0
        max_attempts = len(self.json_data)
        while True:
            json_data = self.json_data[idx]
            latent_pt = torch.load(
                json_data["latent_path"], map_location="cpu", weights_only=True
            )
            T_data = latent_pt["latent"].shape[2]
            if T_data < self.num_training_frames:
                idx = (idx + 1) % len(self.json_data)
                n_attempts += 1
                if n_attempts > max_attempts:
                    raise RuntimeError(
                        f"No segment in dataset has >= {self.num_training_frames} latent frames"
                    )
                continue

            T = self.num_training_frames

            # ---- prompt / byt5 / vision (+ optional CFG dropout) ----
            prompt_embed = latent_pt["prompt_embeds"][0]
            prompt_mask = latent_pt["prompt_mask"][0]
            vision_states = latent_pt["vision_states"][0]
            byt5_text_states = latent_pt["byt5_text_states"][0]
            byt5_text_mask = latent_pt["byt5_text_mask"][0]
            if self.cfg_rate > 0 and self.rng.random() < self.cfg_rate:
                if self.neg_prompt_pt is not None:
                    prompt_embed = self.neg_prompt_pt["negative_prompt_embeds"][0]
                    prompt_mask = self.neg_prompt_pt["negative_prompt_mask"][0]
                if self.neg_byt5_pt is not None:
                    byt5_text_states = self.neg_byt5_pt["byt5_text_states"][0]
                    byt5_text_mask = self.neg_byt5_pt["byt5_text_mask"][0]

            # ---- image_cond + cond_latents (with i2v mask channel) ----
            image_cond_first = latent_pt["image_cond"][0]  # [C, 1, H, W]
            C, _, H, W = image_cond_first.shape
            cond_latents_full = torch.zeros((C + 1, T, H, W), dtype=image_cond_first.dtype)
            cond_latents_full[:C, 0:1] = image_cond_first
            cond_latents_full[C : C + 1, 0:1] = 1.0  # i2v mask at frame 0

            # ---- pose: first T entries (chronological) ----
            pose_json = json.load(open(json_data["pose_path"], "r"))
            pose_keys = list(pose_json.keys())
            intrinsic_list = []
            w2c_list = []
            for i in range(T):
                t_key = pose_keys[0] if i == 0 else pose_keys[4 * (i - 1) + 4]
                intrinsic = np.array(pose_json[t_key]["intrinsic"])
                w2c = np.array(pose_json[t_key]["w2c"])
                intrinsic[0, 0] /= intrinsic[0, 2] * 2
                intrinsic[1, 1] /= intrinsic[1, 2] * 2
                intrinsic[0, 2] = 0.5
                intrinsic[1, 2] = 0.5
                w2c_list.append(w2c)
                intrinsic_list.append(intrinsic)
            w2c_arr = np.array(w2c_list)
            w2c_arr = _camera_center_normalization(w2c_arr)
            intrinsic_tensor = torch.tensor(np.array(intrinsic_list))
            w2c_tensor = torch.tensor(w2c_arr)

            # ---- action: derived from pose via pose_to_input ----
            # Single source of truth. Matches AR dataset fix in
            # trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py:526-542.
            # Bypasses action.json (GameFactory keyboard with A/D direction
            # bug) and the in-loader pose-derive (different threshold).
            # Uses RAW (un-normalized) w2c — pose_to_input only looks at
            # relative_c2w[1:] = inv(c2w[i-1]) @ c2w[i], invariant to global
            # transform, so raw vs center-normalized gives identical labels.
            from hyvideo.generate import pose_to_input
            fake_pose = {}
            for i in range(T):
                t_key = pose_keys[0] if i == 0 else pose_keys[4 * (i - 1) + 4]
                w2c_raw = np.array(pose_json[t_key]["w2c"])
                intrinsic_raw = np.array(pose_json[t_key]["intrinsic"])
                fake_pose[str(i)] = {
                    "extrinsic": np.linalg.inv(w2c_raw).tolist(),
                    "K": intrinsic_raw.tolist(),
                }
            _, _, action_for_pe = pose_to_input(fake_pose, latent_num=T)

            # DEBUG (one-shot, idx==0 in this worker): confirm pose-derived
            # action label is being used + show legacy action.json label for
            # comparison. Remove after smoke-test.
            if idx == 0 and not getattr(self, "_dbg_action_printed", False):
                self._dbg_action_printed = True
                sample_name = os.path.basename(json_data.get("latent_path", "?")).replace("_latent.pt", "")
                print(f"[DMD action DEBUG] sample={sample_name}")
                print(f"[DMD action DEBUG] new (pose_to_input) [:16]: {action_for_pe[:16].tolist()}")
                if "action_path" in json_data:
                    legacy_action = self._read_action_for_pe(json_data["action_path"], T)
                    print(f"[DMD action DEBUG] old (action.json)   [:16]: {legacy_action[:16].tolist()}")
                    n_mis = int((action_for_pe.numpy() != legacy_action.numpy()).sum())
                    print(f"[DMD action DEBUG] mismatched frames vs legacy: {n_mis}/{T}")

            return {
                "cond_latents": cond_latents_full,    # [C+1, T, H, W]
                "image_cond": image_cond_first,        # [C, 1, H, W]
                "w2c": w2c_tensor,                     # [T, 4, 4]
                "intrinsic": intrinsic_tensor,         # [T, 3, 3]
                "action": action_for_pe,               # [T]
                "prompt_embed": prompt_embed,
                "prompt_mask": prompt_mask,
                "vision_states": vision_states,
                "byt5_text_states": byt5_text_states,
                "byt5_text_mask": byt5_text_mask,
                "video_path": json_data.get("video_path", ""),
            }


def build_dmd_dataloader(
    json_path: str,
    num_training_frames: int,
    batch_size: int,
    num_workers: int,
    cfg_rate: float = 0.0,
    seed: int = 0,
    drop_last: bool = True,
):
    """Simple DataLoader builder. No SP-aware sampling — single-GPU smoke first;
    extend with `DP_SP_BatchSampler` once we wire FSDP/SP for multi-GPU."""
    from torch.utils.data import DataLoader, RandomSampler

    dataset = DMDDataset(
        json_path=json_path,
        num_training_frames=num_training_frames,
        cfg_rate=cfg_rate,
        seed=seed,
    )
    sampler = RandomSampler(
        dataset, generator=torch.Generator().manual_seed(seed)
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True,
    )
    return dataset, loader
