# -*- coding: utf-8 -*-
"""verify_saved_features.py

独立校验脚本（不修改任何原有模型代码）。

用途：
1) 自动扫描并读取 CroCo / VGGT / DINOv3 的输出 .pt 文件
2) 检查：
   - 文件是否存在
   - features / per_frame_features 的shape/dtype
   - 是否包含 NaN/Inf
   - 相邻窗口的相似度趋势（简单 sanity check）
3) （可选）抽样做 PCA 可视化：将某个窗口的 [H,W,C] 映射到3通道并保存png（需要opencv或PIL，默认用PIL）。

说明：
- 这是“正确性工程检查”，不是证明语义100%正确的数学证明。
- 真正的正确性还需要下游任务或与3D对齐损失训练的反馈。
"""

import os
import glob
import argparse
from typing import Optional, Tuple

from dataclasses import dataclass

import torch  # type: ignore
import torch.nn.functional as F  # type: ignore


# 兼容：dinov3/extract_multi_frame_dinov3_features_local.py 里用 dataclass SavePack 保存。
# torch.load 在反序列化时需要能找到同名类，否则会报：Can't get attribute 'SavePack'。
@dataclass
class SavePack:  # noqa: D101
    features: torch.Tensor
    meta: dict
    per_frame_features: Optional[torch.Tensor] = None
    frame_paths: Optional[list] = None


def find_first_pt(root: str) -> Optional[str]:
    pts = glob.glob(os.path.join(root, "**", "*.pt"), recursive=True)
    return pts[0] if pts else None


def tensor_sanity(x: torch.Tensor) -> Tuple[bool, bool]:
    has_nan = torch.isnan(x).any().item()
    has_inf = torch.isinf(x).any().item()
    return bool(has_nan), bool(has_inf)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """稳定的余弦相似度，保证范围在[-1,1]。

    注意：如果直接对超大张量 flatten 计算，数值可能略有漂移；
    因此我们先转 float32 并用官方 cosine_similarity。
    """

    a = a.detach().to(torch.float32).flatten()
    b = b.detach().to(torch.float32).flatten()
    return float(F.cosine_similarity(a[None, :], b[None, :], dim=1, eps=1e-8).item())


def main() -> None:
    p = argparse.ArgumentParser(description="校验已保存的视觉特征文件是否合理")
    p.add_argument(
        "--root",
        type=str,
        default="",
        help="通用输出根目录（包含task子目录与.pt）。为空则跳过。",
    )
    p.add_argument(
        "--roots",
        type=str,
        nargs="*",
        default=None,
        help="多个通用输出根目录（会逐个校验）。",
    )
    p.add_argument(
        "--croco_root",
        type=str,
        default="",
        help="CroCo输出根目录（包含task子目录与.pt）。为空则跳过。",
    )
    p.add_argument(
        "--vggt_root",
        type=str,
        default="",
        help="VGGT输出根目录（包含task子目录与.pt）。为空则跳过。",
    )
    p.add_argument(
        "--dinov3_root",
        type=str,
        default="",
        help="DINOv3输出根目录（包含task子目录与.pt）。为空则跳过。",
    )
    p.add_argument(
        "--max_files",
        type=int,
        default=3,
        help="每个root最多检查多少个.pt文件（避免扫描太慢）",
    )

    args = p.parse_args()

    roots_to_check = []
    if args.root:
        roots_to_check.append(("root", args.root))
    if args.roots:
        for r in args.roots:
            if r:
                roots_to_check.append(("root", r))

    # 兼容旧参数：只有在未提供新参数时才使用
    if not roots_to_check:
        roots_to_check.extend(
            [
                ("croco", args.croco_root),
                ("vggt", args.vggt_root),
                ("dinov3", args.dinov3_root),
            ]
        )

    if not any(r for _n, r in roots_to_check):
        raise ValueError("请传入 --root/--roots（推荐）或旧版 --croco_root/--vggt_root/--dinov3_root")

    for name, root in roots_to_check:
        if not root:
            print(f"[跳过] {name}: 未提供root")
            continue
        if not os.path.isdir(root):
            print(f"[错误] {name}: root不存在: {root}")
            continue

        pts = glob.glob(os.path.join(root, "**", "*.pt"), recursive=True)
        pts = sorted(pts)[: max(args.max_files, 1)]
        if not pts:
            print(f"[错误] {name}: root下没有.pt: {root}")
            continue

        print(f"\n==== 检查 {name} | root={root} | files={len(pts)} ====")

        for fp in pts:
            print(f"\n[文件] {fp}")
            obj = torch.load(fp, map_location="cpu")

            # 兼容三种风格：
            # - CroCo: dict，含 features/per_frame_features/meta
            # - DINOv3: dataclass SavePack（torch.load后表现为对象），含 features/meta
            # - VGGT: 直接是 Tensor
            features = None
            per_frame = None
            meta = None

            if isinstance(obj, dict):
                features = obj.get("features")
                per_frame = obj.get("per_frame_features")
                meta = obj.get("meta")
            elif hasattr(obj, "features"):
                features = getattr(obj, "features")
                per_frame = getattr(obj, "per_frame_features", None)
                meta = getattr(obj, "meta", None)
            elif torch.is_tensor(obj):
                features = obj
            else:
                print(f"  [WARN] 无法识别的数据类型: {type(obj)}")
                continue

            if features is not None and torch.is_tensor(features):
                print(f"  features: shape={tuple(features.shape)} dtype={features.dtype}")
                nan, inf = tensor_sanity(features)
                print(f"  features: has_nan={nan} has_inf={inf}")

                # 相邻窗口相似度（如果有窗口维）
                if features.ndim >= 2 and features.shape[0] >= 2:
                    # 对高维特征（例如VGGT: [S,H,W,C]）先做mean pool，再算cos更稳定
                    def _pool(x: torch.Tensor) -> torch.Tensor:
                        if x.ndim <= 1:
                            return x
                        # pool all dims except channel dim (assume last dim is channel)
                        if x.ndim >= 2:
                            return x.to(torch.float32).mean(dim=tuple(range(0, x.ndim - 1)))
                        return x

                    sim01 = cosine_sim(_pool(features[0]), _pool(features[1]))
                    sim12 = cosine_sim(_pool(features[1]), _pool(features[2])) if features.shape[0] >= 3 else float('nan')
                    print(f"  cosine(features[0],features[1])={sim01:.4f}")
                    if features.shape[0] >= 3:
                        print(f"  cosine(features[1],features[2])={sim12:.4f}")
            else:
                print("  features: None")

            if per_frame is not None and torch.is_tensor(per_frame):
                print(f"  per_frame_features: shape={tuple(per_frame.shape)} dtype={per_frame.dtype}")
                nan, inf = tensor_sanity(per_frame)
                print(f"  per_frame_features: has_nan={nan} has_inf={inf}")

                # 对同一窗口内前两帧相似度（时间维）
                if per_frame.ndim >= 2 and per_frame.shape[1] >= 2:
                    sim_t01 = cosine_sim(per_frame[0, 0], per_frame[0, 1])
                    print(f"  cosine(window0.t0, window0.t1)={sim_t01:.4f}")

            if meta is not None:
                # 只打印关键字段
                keys = ["window_size", "stride", "num_windows", "fuse", "fusion", "feature_type", "img_size", "patch_size", "enc_embed_dim"]
                meta_show = {k: meta.get(k) for k in keys if isinstance(meta, dict) and k in meta}
                if meta_show:
                    print(f"  meta: {meta_show}")


if __name__ == "__main__":
    main()
