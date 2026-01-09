"""tools/inspect_pointcloud_sample.py

快速检查点云样本：
- 点数
- xyz 范围
- 是否包含 rgb

用于确认对齐前的数据形态。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# 允许从任意工作目录运行：先把 repo root 加入 sys.path，再导入内部模块
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from features_common.pointcloud import read_ascii_ply_xyz_rgb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ply",
        type=str,
        default="/home/gl/features_model/pc_dataset/PC/beat_block_hammer-demo_randomized-50_head_camera/episode_0/step_0000.ply",
    )
    args = ap.parse_args()

    pack = read_ascii_ply_xyz_rgb(args.ply)
    xyz = pack.xyz

    print("path:", pack.meta.get("path"))
    print("num_points:", xyz.shape[0])
    print("xyz dtype:", xyz.dtype)
    print("xyz min:", np.min(xyz, axis=0))
    print("xyz max:", np.max(xyz, axis=0))

    if pack.rgb is None:
        print("rgb: None")
    else:
        rgb = pack.rgb
        print("rgb dtype:", rgb.dtype)
        print("rgb min:", np.min(rgb, axis=0))
        print("rgb max:", np.max(rgb, axis=0))

    if pack.uv is None:
        print("uv: None (当前 ply 没有像素对应信息)")


if __name__ == "__main__":
    main()
