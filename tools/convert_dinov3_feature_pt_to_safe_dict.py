"""tools/convert_dinov3_feature_pt_to_safe_dict.py

把旧版 dinov3 特征文件（可能是 __main__.SavePack / dataclass）转换成“纯 dict + tensors”的安全格式。

动机
- PyTorch 2.6+ 默认 weights_only=True，会拒绝反序列化自定义类（例如 SavePack）。
- 跨项目使用特征时，不希望每个项目都打 patch 才能 torch.load。

输入
- --in_root:  特征根目录（形如 .../features_dinov3_encoder_dict_unified）
  结构: <in_root>/<task>/episode_0.pt
- --tasks: 可选，限制处理哪些 task

输出
- --out_root: 新的特征根目录（建议 .../features_dinov3_encoder_dict_unified_safe）
  结构保持一致: <out_root>/<task>/episode_0.pt

输出文件 schema（纯 dict）
- per_frame_features: Tensor
- frame_paths: list[list[str]]
- meta: dict
- features: (optional)

可选校验
- --verify: 转换后立即 reload 并检查 shape 与关键字段存在

注意
- 我们默认这些特征是你本地生成、可信来源，因此读取旧格式时会使用 weights_only=False + safe_globals。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch

import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from features_common.feature_pack import load_feature_pack


def save_safe_dict(in_pt: Path, out_pt: Path) -> dict[str, Any]:
    pack = load_feature_pack(in_pt, map_location="cpu")

    d: dict[str, Any] = {
        "meta": dict(pack.meta) if isinstance(pack.meta, dict) else {},
    }
    if pack.per_frame_features is not None:
        d["per_frame_features"] = pack.per_frame_features.cpu()
    if pack.frame_paths is not None:
        d["frame_paths"] = pack.frame_paths
    if pack.features is not None:
        d["features"] = pack.features.cpu()

    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, out_pt)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--tasks", type=str, nargs="*", default=[])
    ap.add_argument("--episodes", type=int, default=None, help="可选：限制 episode 数（例如 20）。")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument(
        "--delete_input",
        action="store_true",
        help="方案2：转换成功（并可选 verify 通过）后删除输入 .pt，节省磁盘空间。",
    )

    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    if not in_root.exists():
        raise FileNotFoundError(f"in_root 不存在: {in_root}")

    tasks = [Path(t) for t in args.tasks] if args.tasks else [p for p in sorted(in_root.iterdir()) if p.is_dir()]

    n_ok = 0
    for task_dir in tasks:
        task_name = task_dir.name if task_dir.is_absolute() is False else task_dir.name
        src_task_dir = in_root / task_name
        if not src_task_dir.exists():
            # 如果用户传的是完整路径 task_dir，则用它
            if task_dir.is_dir() and (task_dir.parent == in_root):
                src_task_dir = task_dir
                task_name = task_dir.name
            else:
                raise FileNotFoundError(f"找不到 task 目录: {task_name} (in_root={in_root})")

        eps = sorted(src_task_dir.glob("episode_*.pt"))
        if args.episodes is not None:
            eps = eps[: max(int(args.episodes), 0)]

        for in_pt in eps:
            out_pt = out_root / task_name / in_pt.name
            try:
                _ = save_safe_dict(in_pt, out_pt)

                if args.verify:
                    obj = torch.load(out_pt, map_location="cpu")
                    assert isinstance(obj, dict)
                    assert "meta" in obj
                    assert "per_frame_features" in obj
                    _ = tuple(obj["per_frame_features"].shape)

                if args.delete_input:
                    in_pt.unlink()

                n_ok += 1
            except Exception:
                # torch.save 可能在磁盘满/写入中断时留下损坏的输出文件；清理后便于重试。
                if out_pt.exists():
                    try:
                        out_pt.unlink()
                    except OSError:
                        pass
                raise

    print(f"Done. Converted {n_ok} files to {out_root}.")


if __name__ == "__main__":
    main()
