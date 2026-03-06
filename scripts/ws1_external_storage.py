#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备 DP2DP3 / features_model 的 ws1 外置存储目录与软链接（默认不影响 baseline）。

你现在的约束是：ws1 视觉特征必须写到外置盘：
  /media/gl/新加卷/gllll/features_model_ws1

本脚本做两件事：
1) 检查外置盘路径是否存在且可写；
2) 创建（或修复）features_model/rgb_dataset_ws1 下的四个软链接：
   - features_{croco,vggt,dinov3,da3}_encoder_dict_unified_zarr -> 外置盘对应目录

可选（默认关闭，避免破坏原始内容）：
- 把 teacher 的 ULIP point tokens、离线特征数据集也迁移到外置盘并建立软链接。

用法示例：
  cd /home/gl/RoboTwin/policy/DP2DP3/features_model
  python scripts/ws1_external_storage.py
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


DEFAULT_EXTERNAL_ROOT = Path("/media/gl/新加卷/gllll/features_model_ws1")


def _is_writable_dir(p: Path) -> bool:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        return False
    return os.access(str(p), os.W_OK)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_symlink(src: Path, dst: Path) -> None:
    """Ensure dst is a symlink pointing to src. Does not overwrite real directories."""
    if dst.is_symlink():
        cur = Path(os.readlink(dst))
        if cur == src:
            return
        dst.unlink()
    elif dst.exists():
        raise RuntimeError(f"Refuse to replace existing path (not symlink): {dst}")
    dst.symlink_to(src)


def _migrate_and_link(local_dir: Path, external_dir: Path) -> None:
    """Move local_dir to external_dir and link back. Safe-ish: requires local_dir to be a real dir."""
    if local_dir.is_symlink():
        # already linked; just ensure target exists
        return
    if not local_dir.exists():
        # nothing to migrate; just link
        _ensure_dir(external_dir)
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        _safe_symlink(external_dir, local_dir)
        return
    if not local_dir.is_dir():
        raise RuntimeError(f"local_dir is not a directory: {local_dir}")
    if external_dir.exists() and any(external_dir.iterdir()):
        raise RuntimeError(f"external_dir already exists and is non-empty: {external_dir}")
    _ensure_dir(external_dir.parent)
    shutil.move(str(local_dir), str(external_dir))
    _safe_symlink(external_dir, local_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--external_root", type=str, default=str(DEFAULT_EXTERNAL_ROOT), help="外置盘根目录")
    ap.add_argument(
        "--migrate_ulip_tokens",
        action="store_true",
        help="把 pc_dataset/ulip_point_tokens_zarr 迁移到外置盘并软链回来（默认关闭）",
    )
    ap.add_argument(
        "--migrate_offline_features",
        action="store_true",
        help="把 data/offline_features_dual_stream_tokens_full_multitask 迁移到外置盘并软链回来（默认关闭）",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]  # features_model/
    external_root = Path(args.external_root).expanduser()

    if not external_root.exists():
        raise SystemExit(
            f"外置盘路径不存在：{external_root}\n"
            "请先确保磁盘已挂载到该路径，并保证当前用户可写（chmod/chown/ACL）。"
        )
    if not _is_writable_dir(external_root):
        raise SystemExit(
            f"外置盘路径不可写：{external_root}\n"
            "请检查挂载与权限（当前用户需要对该目录有写权限）。"
        )

    # 1) ws1 RGB feature roots (must exist on external)
    feature_roots = {
        "croco": external_root / "features_croco_encoder_dict_unified_zarr",
        "vggt": external_root / "features_vggt_encoder_dict_unified_zarr",
        "dinov3": external_root / "features_dinov3_encoder_dict_unified_zarr",
        "da3": external_root / "features_da3_encoder_dict_unified_zarr",
    }
    for p in feature_roots.values():
        _ensure_dir(p)

    # 2) symlinks under rgb_dataset_ws1/
    rgb_ws1_dir = repo_root / "rgb_dataset_ws1"
    rgb_ws1_dir.mkdir(parents=True, exist_ok=True)
    for name, src in feature_roots.items():
        link_name = f"features_{name}_encoder_dict_unified_zarr"
        dst = rgb_ws1_dir / link_name
        _safe_symlink(src, dst)

    # 3) optional migrations
    if bool(args.migrate_ulip_tokens):
        local = repo_root / "pc_dataset" / "ulip_point_tokens_zarr"
        ext = external_root / "ulip_point_tokens_zarr"
        _migrate_and_link(local, ext)

    if bool(args.migrate_offline_features):
        local = repo_root / "data" / "offline_features_dual_stream_tokens_full_multitask"
        ext = external_root / "offline_features_dual_stream_tokens_full_multitask"
        _migrate_and_link(local, ext)

    print("✅ ws1 外置存储准备完成")
    print(f"- external_root: {external_root}")
    print(f"- rgb_dataset_ws1: {rgb_ws1_dir}")


if __name__ == "__main__":
    main()

