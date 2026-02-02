#!/usr/bin/env python3
"""
提取 ULIP PointBERT 的局部 token 特征并保存为 step 级 zarr。
输出格式（每个 step 一个 zarr group）：
- tokens: [G, D]  (G=group数, D=trans_dim)
- centers: [G, 3]

用途：为 RGB token 对齐提供局部 teacher 监督。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import zarr
from tqdm import tqdm

# 固定随机种子
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

WORKSPACE_DIR = "/home/gl/RoboTwin/policy/DP2DP3/features_model"
ULIP_DIR = os.path.join(WORKSPACE_DIR, "ULIP-main/ULIP-main")
PC_SOURCE_DIR = os.path.join(WORKSPACE_DIR, "pc_dataset/PC_ORI")

# ====== ULIP imports ======
import sys
sys.path.append(str(Path(ULIP_DIR).absolute()))
original_cwd = os.getcwd()
os.chdir(str(Path(ULIP_DIR).absolute()))

try:
    from models.ULIP_models import ULIP2_WITH_OPENCLIP
    from models.pointbert.point_encoder import PointTransformer_Colored
    from utils.config import cfg_from_yaml_file
except Exception as e:
    print("❌ Failed to import ULIP dependencies:")
    print(repr(e))
    sys.exit(1)


class PointTransformerColoredWithTokens(PointTransformer_Colored):
    """保持原有forward不变，新增forward_tokens返回group token与中心点。"""

    def forward_tokens(self, pts: torch.Tensor):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)
        group_input_tokens = self.reduce_dim(group_input_tokens)
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        pos = self.pos_embed(center)
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        x = self.blocks(x, pos)
        x = self.norm(x)
        # x: [B, 1+G, D]
        return x[:, 1:], center


def load_ulip_point_encoder(device: torch.device):
    config_path = os.path.join(ULIP_DIR, "models/pointbert/ULIP_2_PointBERT_10k_colored_pointclouds.yaml")
    config = cfg_from_yaml_file(config_path)

    class Args:
        evaluate_3d = True

    point_encoder = PointTransformerColoredWithTokens(config.model, args=Args())
    model = ULIP2_WITH_OPENCLIP(point_encoder=point_encoder, open_clip_model=None, pc_feat_dims=768)

    ckpt_path = os.path.join(
        ULIP_DIR,
        "pretrain_model/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt",
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"✅ ULIP Load Success | Missing: {len(msg.missing_keys)} | Unexpected: {len(msg.unexpected_keys)}")

    model = model.to(device).eval()
    return model


def project_tokens_to_1280(tokens_384: torch.Tensor, cls_token: torch.Tensor, pc_projection: torch.Tensor) -> torch.Tensor:
    """使用ULIP的pc_projection把token投影到1280维。

    tokens_384: [B, G, 384]
    cls_token: [B, 1, 384]
    pc_projection: [768, 1280]
    """
    cls_rep = cls_token.expand(-1, tokens_384.shape[1], -1)
    tokens_768 = torch.cat([cls_rep, tokens_384], dim=-1)
    return tokens_768 @ pc_projection


def read_ply(ply_path: str):
    ply_path = str(Path(ply_path).absolute())
    valid_xyz = np.zeros((0, 3), dtype=np.float32)
    valid_rgb = np.zeros((0, 3), dtype=np.float32)

    with open(ply_path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                break
            header_lines.append(line)
            if line.strip() == b"end_header":
                break

        data_start = f.tell()
        header_txt = [ln.decode("ascii", errors="ignore").strip() for ln in header_lines]
        if not header_txt or header_txt[0] != "ply":
            raise ValueError("Not a valid PLY file")

        fmt, vertex_count, in_vertex = None, 0, False
        properties = []
        for ln in header_txt:
            if ln.startswith("format "):
                fmt = ln.split()[1]
            elif ln.startswith("element vertex"):
                vertex_count = int(ln.split()[-1]) if len(ln.split()) > 2 else 0
                in_vertex = True
            elif ln.startswith("element "):
                in_vertex = False
            elif in_vertex and ln.startswith("property ") and "list" not in ln:
                properties.append((ln.split()[1], ln.split()[2]))

        prop_names = [n for _, n in properties]
        if not all(k in prop_names for k in ("x", "y", "z")):
            raise ValueError("Missing XYZ coordinates")

        rgb_keys = None
        if all(k in prop_names for k in ("red", "green", "blue")):
            rgb_keys = ("red", "green", "blue")
        elif all(k in prop_names for k in ("r", "g", "b")):
            rgb_keys = ("r", "g", "b")

        if fmt == "ascii" and vertex_count > 0:
            f.seek(data_start)
            valid_data = []
            for _ in range(vertex_count):
                line = f.readline()
                if not line:
                    break
                parts = line.decode("ascii", errors="ignore").strip().split()
                if len(parts) >= len(properties):
                    valid_data.append(parts)
            vertex_count = len(valid_data)
            valid_xyz = np.zeros((vertex_count, 3), dtype=np.float32)
            valid_rgb = np.zeros((vertex_count, 3), dtype=np.float32)

            for i, parts in enumerate(valid_data):
                prop_vals = {n: float(v) for (t, n), v in zip(properties, parts)}
                valid_xyz[i] = [prop_vals["x"], prop_vals["y"], prop_vals["z"]]
                if rgb_keys:
                    valid_rgb[i] = [prop_vals[rgb_keys[0]], prop_vals[rgb_keys[1]], prop_vals[rgb_keys[2]]]

    if rgb_keys and valid_rgb.size > 0 and valid_rgb.max() > 1.0:
        valid_rgb = valid_rgb / 255.0
    valid_rgb = np.clip(valid_rgb, 0.0, 1.0)

    if valid_xyz.shape[0] == 0:
        return np.zeros((1, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.float32)
    return valid_xyz, valid_rgb


def normalize_pc(xyz: np.ndarray) -> np.ndarray:
    if len(xyz) <= 1:
        return xyz.copy()
    centroid = np.mean(xyz, axis=0)
    xyz_centered = xyz - centroid
    max_dist = np.max(np.sqrt(np.sum(xyz_centered**2, axis=1)))
    if max_dist < 1e-6:
        return xyz_centered
    return xyz_centered / (max_dist + 1e-6)


def fps_sample(xyz: np.ndarray, n_points: int = 2048):
    N = xyz.shape[0]
    if N <= n_points:
        idx = np.arange(N, dtype=np.int64)
        if N < n_points and N > 0:
            pad_idx = np.random.choice(N, n_points - N, replace=True)
            idx = np.concatenate([idx, pad_idx], axis=0)
        return idx

    centroids = np.zeros(n_points, dtype=np.int64)
    dist = np.ones(N, dtype=np.float32) * 1e10
    centroids[0] = np.random.randint(0, N)
    for i in range(1, n_points):
        dist_i = np.sum((xyz - xyz[centroids[i - 1]]) ** 2, axis=1)
        dist = np.minimum(dist, dist_i)
        centroids[i] = np.argmax(dist)
    return centroids


def sample_pc(xyz: np.ndarray, rgb: np.ndarray, n_points: int = 2048):
    if xyz.shape[0] == 0:
        return np.zeros((n_points, 3), dtype=np.float32), np.zeros((n_points, 3), dtype=np.float32)
    idx = fps_sample(xyz, n_points)
    return xyz[idx], rgb[idx]


def main():
    parser = argparse.ArgumentParser(description="ULIP per-point token提取（step级 zarr）")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(WORKSPACE_DIR, "pc_dataset/ulip_point_tokens_zarr"),
        help="输出根目录",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的step zarr")
    parser.add_argument("--task", type=str, default=None, help="仅处理指定task")
    parser.add_argument("--episodes", type=int, default=0, help="仅处理前N个episode(0表示全部)")
    parser.add_argument("--max_steps", type=int, default=0, help="每个episode最多处理step数(0表示全部)")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    overwrite = bool(args.overwrite)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n✅ Start ULIP token extraction | Device: {device}")
    model = load_ulip_point_encoder(device)

    if not os.path.exists(PC_SOURCE_DIR):
        print(f"❌ Source Dir Not Found: {PC_SOURCE_DIR}")
        return

    if args.task:
        tasks = [args.task]
    else:
        tasks = [d for d in os.listdir(PC_SOURCE_DIR) if os.path.isdir(os.path.join(PC_SOURCE_DIR, d))]

    for task in tqdm(tasks, desc="📌 All Tasks", ncols=80):
        task_dir = os.path.join(PC_SOURCE_DIR, task)
        out_task_dir = output_root / task
        out_task_dir.mkdir(parents=True, exist_ok=True)

        episodes = [d for d in os.listdir(task_dir) if d.startswith("episode_")]
        episodes = sorted(episodes)
        if int(args.episodes) > 0:
            episodes = episodes[: int(args.episodes)]
        for episode in tqdm(episodes, desc=f"📌 {task} Episodes", ncols=80, leave=False):
            episode_dir = os.path.join(task_dir, episode)
            out_ep_dir = out_task_dir / episode
            out_ep_dir.mkdir(parents=True, exist_ok=True)

            ply_files = sorted(glob.glob(os.path.join(episode_dir, "*.ply")))
            if int(args.max_steps) > 0:
                ply_files = ply_files[: int(args.max_steps)]
            if not ply_files:
                continue

            for ply_file in ply_files:
                stem = Path(ply_file).name
                out_path = out_ep_dir / f"{stem}.ulip_tokens.zarr"
                if out_path.exists():
                    if not overwrite:
                        continue
                    shutil.rmtree(out_path, ignore_errors=True)

                try:
                    xyz, rgb = read_ply(ply_file)
                    xyz = normalize_pc(xyz)
                    xyz, rgb = sample_pc(xyz, rgb, 2048)
                    pc = np.concatenate([xyz, rgb], axis=1)
                    pc_tensor = torch.from_numpy(pc).float().unsqueeze(0).to(device)

                    with torch.no_grad():
                        tokens_384, centers = model.point_encoder.forward_tokens(pc_tensor)
                        cls_token = model.point_encoder.cls_token.expand(tokens_384.size(0), -1, -1)
                        tokens_1280 = project_tokens_to_1280(
                            tokens_384,
                            cls_token,
                            model.pc_projection,
                        )
                    tokens_np = tokens_1280.squeeze(0).cpu().numpy().astype(np.float32)
                    centers_np = centers.squeeze(0).cpu().numpy().astype(np.float32)

                    g = zarr.group(str(out_path), overwrite=True)
                    g.create_dataset("tokens", data=tokens_np, chunks=(min(256, tokens_np.shape[0]), tokens_np.shape[1]), dtype="float32")
                    g.create_dataset("centers", data=centers_np, chunks=(min(256, centers_np.shape[0]), 3), dtype="float32")

                    meta = {
                        "task": task,
                        "episode": episode,
                        "step": stem,
                        "token_dim": int(tokens_np.shape[1]),
                        "num_tokens": int(tokens_np.shape[0]),
                        "source_point_num": 2048,
                    }
                    with open(out_ep_dir / f"{stem}.ulip_tokens.meta.json", "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"⚠️ Skip {stem}: {e}")
                    continue


if __name__ == "__main__":
    main()
