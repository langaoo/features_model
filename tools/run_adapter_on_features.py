"""tools/run_adapter_on_features.py

把某个模型导出的 features_root 跑一遍 adapter，生成“统一维度”的输出。用于：
- 后续跨模型对齐训练前的预处理/缓存
- 或者快速验证 adapter 的输出没有 NaN/Inf

输出
- 默认保存为 dict：
  - per_frame_features: [W,T,Hf,Wf,D]
  - features: [W,Hf,Wf,D]   （这是 adapter 的时间聚合结果，不是手工 mean）
  - frame_paths/meta 原样复制并追加 adapter 配置

注意
- 这是离线处理脚本，不涉及任何点云/几何。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch  # type: ignore
from tqdm import tqdm

# 允许从任意工作目录直接运行：把仓库根目录加入 sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from features_common.adapters import MultiModelAdapter
from features_common.feature_pack import load_feature_pack


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--task", type=str, default=None)
    ap.add_argument("--episodes", type=str, nargs="*", default=None, help="指定 episode 名称列表（例如 episode_0）")

    ap.add_argument("--out_dim", type=int, default=256)
    ap.add_argument("--no_temporal_pool", action="store_true")

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])

    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    save_dtype = dtype_map[args.save_dtype]

    # 找到所有 pt
    pts = sorted(in_root.glob("*/*.pt"))
    if args.task is not None:
        pts = [p for p in pts if p.parent.name == args.task]
    if args.episodes is not None:
        wanted = set(args.episodes)
        pts = [p for p in pts if p.stem in wanted]

    if not pts:
        raise FileNotFoundError("in_root 下没有匹配的 pt 文件")

    # 用第一个样本推断 in_dim
    first = load_feature_pack(pts[0])
    if first.per_frame_features is None:
        raise RuntimeError("第一个样本没有 per_frame_features")
    in_dim = int(first.per_frame_features.shape[-1])

    adapter = MultiModelAdapter(in_dim=in_dim, out_dim=int(args.out_dim), use_temporal_pool=not args.no_temporal_pool)
    adapter = adapter.to(device)
    adapter.eval()

    for pt in tqdm(pts, desc="adapter"):
        pack = load_feature_pack(pt)
        pf = pack.per_frame_features
        if pf is None:
            continue

        x = pf.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            out = adapter(x)

        per_frame = out.per_frame.to(dtype=save_dtype).cpu()
        fused = out.pooled.to(dtype=save_dtype).cpu() if out.pooled is not None else None

        out_obj = {
            "per_frame_features": per_frame,
            "frame_paths": pack.frame_paths,
            "meta": dict(pack.meta),
        }
        out_obj["meta"].update({
            "adapter": {
                "type": "MultiModelAdapter",
                "in_dim": in_dim,
                "out_dim": int(args.out_dim),
                "use_temporal_pool": not args.no_temporal_pool,
            }
        })
        if fused is not None:
            out_obj["features"] = fused

        rel = pt.relative_to(in_root)
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out_obj, dst)


if __name__ == "__main__":
    main()
