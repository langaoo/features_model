"""features_common/feature_pack.py

统一读取四模型特征 .pt 的工具。

为什么需要它？
- 你现在的特征文件可能是三种形态：
  1) dict（推荐：croco/vggt/da3 已是这种）
  2) dataclass（dinov3 早期 SavePack）
  3) 直接 Tensor（vggt 旧脚本可能输出过）

本文件把它们统一成一个 FeaturePack dataclass，方便写 loader / 单元测试。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import types
import sys

import torch  # type: ignore


@dataclass
class FeaturePack:
    """统一后的特征样本。

    - per_frame_features: [W,T,Hf,Wf,C]（推荐，默认应存在）
    - frame_paths: List[List[str]]，长度为 W
    - meta: dict
    - features: （可选）融合后的窗口特征
    """

    per_frame_features: Optional[torch.Tensor]
    frame_paths: Optional[list[list[str]]]
    meta: dict[str, Any]
    features: Optional[torch.Tensor] = None


# -------------------------
# 兼容旧版 dinov3 的 SavePack
# -------------------------
@dataclass
class SavePack:  # noqa: D101
    features: torch.Tensor
    meta: dict
    per_frame_features: Optional[torch.Tensor] = None
    frame_paths: Optional[list] = None


def _ensure_main_has_savepack() -> None:
    """确保反序列化时能找到 __main__.SavePack。

    旧版 dinov3 脚本在 top-level 定义了 SavePack，并用 torch.save 保存其对象。
    反序列化时 pickle 会尝试从 __main__ 找 SavePack。
    pytest/python -c 的 __main__ 不包含该类，所以需要我们临时注入。
    """

    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        main_mod = types.ModuleType("__main__")
        sys.modules["__main__"] = main_mod
    if not hasattr(main_mod, "SavePack"):
        setattr(main_mod, "SavePack", SavePack)


def _as_dict(obj: Any) -> dict[str, Any]:
    """把 torch.load 的对象尽量转成 dict view。"""

    if isinstance(obj, dict):
        return obj

    # 兼容 dinov3 的 dataclass SavePack
    if hasattr(obj, "__dict__"):
        d = dict(obj.__dict__)
        return d

    # 兼容旧 vggt：直接 Tensor
    if torch.is_tensor(obj):
        return {"features": obj, "meta": {"legacy_tensor": True}}

    raise TypeError(f"不支持的 .pt 内容类型: {type(obj)}")


def load_feature_pack(path: str | Path, *, map_location: str = "cpu") -> FeaturePack:
    path = Path(path)
    # PyTorch 2.6 起 torch.load 的默认 weights_only=True，会拒绝反序列化自定义类（例如旧版 dinov3 的 SavePack）。
    # 这里加载的都是你本地生成/信任的特征文件，因此我们显式使用 weights_only=False 保持兼容。
    _ensure_main_has_savepack()
    # 兼容 torch>=2.6 的安全反序列化：需要显式 allowlist 自定义类。
    try:
        from torch.serialization import safe_globals  # type: ignore

        with safe_globals([SavePack]):
            obj = torch.load(str(path), map_location=map_location, weights_only=False)
    except Exception:
        # fallback：老版本 torch 没有 safe_globals，或某些环境下导入失败
        obj = torch.load(str(path), map_location=map_location, weights_only=False)
    d = _as_dict(obj)

    meta = d.get("meta")
    if meta is None or not isinstance(meta, dict):
        meta = {}

    pack = FeaturePack(
        per_frame_features=d.get("per_frame_features"),
        frame_paths=d.get("frame_paths"),
        meta=meta,
        features=d.get("features"),
    )

    # 兜底：某些早期文件只存了 features，但那也能读；只是对齐测试会更严格。
    return pack


def infer_task_episode_from_path(pt_path: str | Path) -> tuple[str, str]:
    """从 .../<task>/<episode>.pt 推断 task/episode。"""
    p = Path(pt_path)
    episode = p.stem
    task = p.parent.name
    return task, episode
