"""features_common/pointcloud.py

点云读取与统一契约。

现状（通过实际读取 step_0000.ply header 确认）：
- 你的点云 ply 格式是 ascii 1.0
- 顶点属性通常是：x,y,z + red,green,blue
- **有些导出会额外包含 (u,v) 像素索引**（本文件已支持解析）

这意味着：
- 如果你要做严格的 point↔pixel/patch 对齐（最推荐），需要每点到像素的对应信息。
    - 最理想：点云里直接带 (u,v)（你现在的 step_0000.ply 就是这种）。
    - 次优：保存当时的深度图 + 相机内参/外参，然后把点投回像素。

本文件先把“点云帧”标准化：
- xyz: [N,3] float32
- rgb: [N,3] uint8 (可选)

后续如果你有 uv，我们可以扩展：
- uv: [N,2] int32
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class PointCloudFramePack:
    xyz: np.ndarray  # [N,3] float32
    rgb: Optional[np.ndarray]  # [N,3] uint8
    meta: dict[str, Any]
    uv: Optional[np.ndarray] = None  # [N,2] int32（当前 ply 不含该字段）


def read_ascii_ply_xyz_rgb(path: str | Path) -> PointCloudFramePack:
    """读取 ascii PLY。

    支持两种常见格式：
    - x y z red green blue
    - x y z red green blue u v

    其中 (u,v) 是每个点对应的源图像像素坐标（通常以像素为单位）。
    """

    path = Path(path)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        header = []
        for _ in range(300):
            line = f.readline()
            if not line:
                break
            header.append(line.strip())
            if header[-1] == "end_header":
                break

        # 粗略解析 vertex 数
        n = None
        for ln in header:
            if ln.startswith("element vertex"):
                n = int(ln.split()[-1])
                break
        if n is None:
            raise ValueError(f"无法从 header 解析 vertex 数量: {path}")

        # 检查 header 是否包含 u/v
        prop_names: list[str] = []
        for ln in header:
            if ln.startswith("property "):
                prop_names.append(ln.split()[-1])
        has_uv = len(prop_names) >= 8 and prop_names[6:8] == ["u", "v"]

        # 读取点：6列或8列
        data = []
        for _ in range(n):
            ln = f.readline()
            if not ln:
                break
            parts = ln.strip().split()
            if len(parts) < (8 if has_uv else 6):
                continue
            data.append(parts[: (8 if has_uv else 6)])

    arr = np.asarray(data, dtype=np.float32)
    if arr.shape[1] not in (6, 8):
        raise ValueError(
            f"PLY 列数不为6或8（x y z r g b [u v]），实际 shape={arr.shape}"
        )

    xyz = arr[:, :3].astype(np.float32)
    rgb_f = arr[:, 3:6]
    # rgb 在文件里是 0..255，但被我们读成 float32，这里转 uint8
    rgb = np.clip(rgb_f, 0, 255).astype(np.uint8)

    uv = None
    if arr.shape[1] == 8:
        # 这里保留 float32（例如 0.0, 1.0, ...），后续映射到 patch 前再做 round/clip
        uv = arr[:, 6:8].astype(np.float32)

    return PointCloudFramePack(
        xyz=xyz,
        rgb=rgb,
        meta={
            "path": str(path),
            "num_points": int(xyz.shape[0]),
            "format": "ply/ascii",
        },
        uv=uv,
    )
