# -*- coding: utf-8 -*-
"""visualize_feature_pca.py

对保存的特征做非常轻量的可视化（PCA->伪彩色）。

支持输入：
- CroCo: dict，优先用 per_frame_features[win_idx, t_idx] 或 features[win_idx]
- VGGT: Tensor，features[win_idx, t_idx]
- DINOv3: 可能是 SavePack(global) -> 无法做空间图，仅打印/跳过

输出：
- 在 --out_dir 下保存 png

注意：
- 这不是“语义证明”，只是 sanity check：如果特征里有结构，PCA图往往能显出边界/区域。
"""

import os
import argparse
from typing import Optional

from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image


# 兼容：dinov3/extract_multi_frame_dinov3_features_local.py 用 dataclass SavePack 保存。
# torch.load 在反序列化时需要同名类，否则会报：Can't get attribute 'SavePack'。
@dataclass
class SavePack:  # noqa: D101
    features: torch.Tensor
    meta: dict
    per_frame_features: Optional[torch.Tensor] = None
    frame_paths: Optional[list] = None


def pca_to_rgb(x_hw_c: torch.Tensor) -> np.ndarray:
    """x: [H,W,C] -> uint8 RGB [H,W,3]"""
    x = x_hw_c.detach().to(torch.float32)
    h, w, c = x.shape
    X = x.reshape(h * w, c).cpu().numpy()
    X = X - X.mean(axis=0, keepdims=True)

    # SVD PCA (no sklearn)
    # U,S,Vt = svd(X); components = Vt[:3]
    try:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        comps = Vt[:3]
        Y = X @ comps.T  # [HW,3]
    except np.linalg.LinAlgError:
        # fallback: random projection
        rng = np.random.default_rng(0)
        W = rng.standard_normal((c, 3)).astype(np.float32)
        Y = X @ W

    Y = Y.reshape(h, w, 3)
    # robust normalize to [0,1]
    lo = np.percentile(Y, 1, axis=(0, 1), keepdims=True)
    hi = np.percentile(Y, 99, axis=(0, 1), keepdims=True)
    Y = (Y - lo) / (hi - lo + 1e-6)
    Y = np.clip(Y, 0, 1)
    return (Y * 255).astype(np.uint8)


def load_any(pt_path: str):
    obj = torch.load(pt_path, map_location="cpu")
    return obj


def infer_tag(obj, pt_path: str) -> str:
    """推断输出前缀，避免不同模型的 episode_0 互相覆盖。"""

    # dict: CroCo / VGGT wrapper
    if isinstance(obj, dict):
        meta = obj.get("meta", {})
        if isinstance(meta, dict) and meta.get("model"):
            return str(meta.get("model"))
        return "dict"

    # SavePack
    if hasattr(obj, "meta"):
        meta = getattr(obj, "meta", None)
        if isinstance(meta, dict) and meta.get("model"):
            return str(meta.get("model"))
        return "savepack"

    if torch.is_tensor(obj):
        return "tensor"

    return os.path.basename(pt_path).replace(".pt", "")


def pick_map(obj, win_idx: int, t_idx: int) -> Optional[torch.Tensor]:
    # CroCo dict
    if isinstance(obj, dict):
        if obj.get("per_frame_features") is not None:
            pf = obj["per_frame_features"]
            return pf[win_idx, t_idx]
        if obj.get("features") is not None:
            f = obj["features"]
            return f[win_idx]
        return None

    # SavePack/global: no spatial
    if hasattr(obj, "features"):
        pf = getattr(obj, "per_frame_features", None)
        if torch.is_tensor(pf) and pf.ndim == 5:
            # [num_windows, S, H, W, C]
            return pf[win_idx, t_idx]

        feats = getattr(obj, "features")
        if torch.is_tensor(feats) and feats.ndim >= 3:
            # [num_windows, N, C] -> 粗暴reshape不安全，先不做
            return None
        return None

    # VGGT tensor
    if torch.is_tensor(obj):
        f = obj
        # [num_windows, S, H, W, C]
        if f.ndim == 5:
            return f[win_idx, t_idx]
        # [num_windows, H, W, C]
        if f.ndim == 4:
            return f[win_idx]

    return None


def main() -> None:
    p = argparse.ArgumentParser(description="对特征做PCA伪彩色可视化")
    p.add_argument("--pt_path", type=str, required=True, help="某个.pt特征文件路径")
    p.add_argument("--out_dir", type=str, required=True, help="输出目录")
    p.add_argument("--win_idx", type=int, default=0, help="窗口索引")
    p.add_argument("--t_idx", type=int, default=0, help="时间索引（有时间维时）")
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
    if fmap is None:
        print("[跳过] 无法从该文件中提取空间feature map（可能是global特征）")
        return

    if fmap.ndim != 3:
        print(f"[跳过] feature map 维度不是[H,W,C]：{tuple(fmap.shape)}")
        return

    rgb = pca_to_rgb(fmap)
    out_path = os.path.join(args.out_dir, f"pca_{tag}_{os.path.basename(args.pt_path).replace('.pt','')}_w{args.win_idx}_t{args.t_idx}.png")
    Image.fromarray(rgb).save(out_path)
    print(f"[保存] {out_path} | map_shape={tuple(fmap.shape)}")


if __name__ == "__main__":
    main()
