"""features_common/zarr_pack.py

Zarr 版 FeaturePack 随机访问读取。

配合 `tools/convert_episode_pt_to_zarr.py` 生成的目录结构：

<episode>.zarr/
  per_frame_features  (zarr array) shape [W,T,Hf,Wf,C]
  frame_paths.json
  meta.json
  shape.json

目标
- 训练时只读取所需 window/frame，避免 torch.load 整个 episode 导致 CPU/RSS 爆炸。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr


@dataclass
class ZarrPack:
    path: Path
    arr: Any  # zarr.Array
    frame_paths: list[list[str]]
    meta: dict[str, Any]

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        return tuple(self.arr.shape)  # type: ignore[return-value]

    def get_window(self, wi: int) -> np.ndarray:
        """返回 window: [T,Hf,Wf,C] (numpy)"""
        return np.asarray(self.arr[wi])

    def get_frame(self, wi: int, ti: int) -> np.ndarray:
        """返回帧: [Hf,Wf,C] (numpy)"""
        return np.asarray(self.arr[wi, ti])


def load_zarr_pack(path: str | Path) -> ZarrPack:
    p = Path(path)
    store = zarr.DirectoryStore(str(p))
    root = zarr.open(store=store, mode="r")
    arr = root["per_frame_features"]
    # 有些早期/中断转换可能只写了 zarr array 而没写 json 元数据。
    # 为了不让训练直接崩溃，这里做一个温和的降级：
    # - frame_paths.json 缺失：按 shape 生成 step_0000.. 的占位路径
    # - meta.json 缺失：使用空 dict
    fp_json = p / "frame_paths.json"
    meta_json = p / "meta.json"

    if fp_json.exists():
        frame_paths = json.loads(fp_json.read_text(encoding="utf-8"))
    else:
        W, T = int(arr.shape[0]), int(arr.shape[1])
        frame_paths = [[f"step_{wi*T+ti:04d}" for ti in range(T)] for wi in range(W)]
        # 写回去，后续就不会再走降级路径
        try:
            fp_json.write_text(json.dumps(frame_paths), encoding="utf-8")
        except Exception:
            pass

    if meta_json.exists():
        meta = json.loads(meta_json.read_text(encoding="utf-8"))
    else:
        meta = {}
        try:
            meta_json.write_text(json.dumps(meta), encoding="utf-8")
        except Exception:
            pass
    return ZarrPack(path=p, arr=arr, frame_paths=frame_paths, meta=meta)
