# -*- coding: utf-8 -*-
"""visualize_feature_energy.py

把高维特征图做成更“像边界图”的可视化（比 PCA 更直观）。

输入
- --pt_path 指向一个 .pt 特征文件
  - CroCo/VGGT-wrapper: dict，优先用 per_frame_features[win_idx, t_idx] 或 features[win_idx]
  - DINOv3: SavePack，优先用 per_frame_features[win_idx, t_idx]

输出（保存到 --out_dir）
- energy: 每个位置的通道 L2 能量:  E(x,y)=||f(x,y,:)||_2
- grad:   energy 的简单梯度幅值（Sobel-like 的差分版）

命名
- 输出文件名会自动带上来源前缀，避免 "episode_0" 互相覆盖：
  energy_<tag>_w0_t0.png, grad_<tag>_w0_t0.png

注意
- 这不是语义证明；但若特征对局部结构/边界敏感，grad 图通常会更清晰。
"""

from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from PIL import Image


# 兼容：dinov3 保存的 SavePack
@dataclass
class SavePack:  # noqa: D101
    features: torch.Tensor
    meta: dict
    per_frame_features: Optional[torch.Tensor] = None
    frame_paths: Optional[list] = None


def load_any(pt_path: str):
    return torch.load(pt_path, map_location="cpu")


def infer_tag(obj, pt_path: str) -> str:
    # 优先从 meta.model 推断
    if isinstance(obj, dict):
        meta = obj.get("meta", {})
        if isinstance(meta, dict) and meta.get("model"):
            return str(meta.get("model"))
        # vggt wrapper / croco
        if "per_frame_features" in obj and obj.get("meta", {}).get("note", "").startswith("per_frame_features"):
            return "vggt"
        return "dict"

    if hasattr(obj, "meta"):
        meta = getattr(obj, "meta", None)
        if isinstance(meta, dict) and meta.get("model"):
            return str(meta.get("model"))
        return "savepack"

    if torch.is_tensor(obj):
        return "tensor"

    # fallback: 文件名
    return os.path.basename(pt_path).replace(".pt", "")


def pick_map(obj, win_idx: int, t_idx: int) -> Optional[torch.Tensor]:
    # dict: CroCo/VGGT wrapper
    if isinstance(obj, dict):
        if obj.get("per_frame_features") is not None:
            pf = obj["per_frame_features"]
            return pf[win_idx, t_idx]
        if obj.get("features") is not None:
            f = obj["features"]
            # [W,H,W,C]
            if torch.is_tensor(f) and f.ndim == 4:
                return f[win_idx]
        return None

    # SavePack: DINOv3
    if hasattr(obj, "per_frame_features"):
        pf = getattr(obj, "per_frame_features", None)
        if torch.is_tensor(pf) and pf.ndim == 5:
            return pf[win_idx, t_idx]
        return None

    # legacy tensor: VGGT old
    if torch.is_tensor(obj):
        f = obj
        if f.ndim == 5:
            return f[win_idx, t_idx]
        if f.ndim == 4:
            return f[win_idx]

    return None


def norm_to_uint8(x: np.ndarray) -> np.ndarray:
    # robust normalize to [0,255]
    lo = np.percentile(x, 1)
    hi = np.percentile(x, 99)
    y = (x - lo) / (hi - lo + 1e-8)
    y = np.clip(y, 0, 1)
    return (y * 255).astype(np.uint8)


def energy_map(fmap: torch.Tensor) -> np.ndarray:
    # fmap: [H,W,C]
    x = fmap.detach().to(torch.float32)
    e = torch.linalg.vector_norm(x, ord=2, dim=-1)  # [H,W]
    return e.cpu().numpy()


def grad_mag(e: np.ndarray) -> np.ndarray:
    # simple 1st-order difference (Sobel-like but simpler, no cv2 dependency)
    dx = np.zeros_like(e)
    dy = np.zeros_like(e)
    dx[:, 1:] = e[:, 1:] - e[:, :-1]
    dy[1:, :] = e[1:, :] - e[:-1, :]
    g = np.sqrt(dx * dx + dy * dy)
    return g


def save_gray(img_hw: np.ndarray, out_path: str) -> None:
    Image.fromarray(norm_to_uint8(img_hw), mode="L").save(out_path)


def main() -> None:
    p = argparse.ArgumentParser(description="把特征图做能量/边界可视化")
    p.add_argument("--pt_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--win_idx", type=int, default=0)
    p.add_argument("--t_idx", type=int, default=0)
    p.add_argument(
        "--tag",
        type=str,
        default=None,
        help="可选：手工指定输出前缀；不填则从meta/类型自动推断",
    )

    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    obj = load_any(args.pt_path)
    tag = args.tag or infer_tag(obj, args.pt_path)

    fmap = pick_map(obj, win_idx=args.win_idx, t_idx=args.t_idx)
    if fmap is None or fmap.ndim != 3:
        print("[跳过] 无法从该文件中提取[H,W,C] feature map")
        return

    e = energy_map(fmap)
    g = grad_mag(e)

    base = f"{tag}_w{args.win_idx}_t{args.t_idx}"
    out_e = os.path.join(args.out_dir, f"energy_{base}.png")
    out_g = os.path.join(args.out_dir, f"grad_{base}.png")

    save_gray(e, out_e)
    save_gray(g, out_g)

    print(f"[保存] {out_e} | shape={e.shape}")
    print(f"[保存] {out_g} | shape={g.shape}")


if __name__ == "__main__":
    main()
