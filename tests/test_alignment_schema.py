"""跨模型对齐单元测试（schema/窗口/帧路径对齐）。

用法（示例）：
- 你先分别跑四个模型，把输出放在四个 root 目录（每个 root 下是 task/episode.pt）
- 然后运行 pytest，并传入目录环境变量：

  FEATURES_CROCO_ROOT=... \
  FEATURES_VGGT_ROOT=... \
  FEATURES_DINOV3_ROOT=... \
  FEATURES_DA3_ROOT=... \
  pytest -q

为什么用环境变量？
- 不强依赖固定路径，CI/本地都方便。

注意
- 这是“对齐格式”的单元测试，不验证语义是否真的对齐（语义要靠下游损失/指标）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch  # type: ignore

# 让 `pytest` 在任何工作目录下都能 import 到 repo 内的 features_common
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from features_common.feature_pack import load_feature_pack
from features_common.zarr_pack import load_zarr_pack


def _first_episode_file(root: str) -> Path:
    """返回第一个 episode 文件（支持旧 .pt 和新 .zarr）。"""

    p = Path(root)
    pts = sorted(p.glob("*/*.pt"))
    if pts:
        return pts[0]
    zarrs = sorted(p.glob("*/*.zarr"))
    if zarrs:
        return zarrs[0]
    raise FileNotFoundError(f"root 下找不到任何 task/episode.(pt|zarr): {root}")


def _load_pack_any(path: Path):
    if path.suffix == ".pt":
        return load_feature_pack(path)
    if path.suffix == ".zarr":
        return load_zarr_pack(path)
    raise ValueError(f"unsupported feature pack suffix: {path}")


def _require_root(env: str) -> str:
    default_map = {
        "FEATURES_CROCO_ROOT": "/home/gl/features_model/rgb_dataset/features_croco_encoder_dict_unified_zarr",
        "FEATURES_VGGT_ROOT": "/home/gl/features_model/rgb_dataset/features_vggt_encoder_dict_unified_zarr",
        "FEATURES_DINOV3_ROOT": "/home/gl/features_model/rgb_dataset/features_dinov3_encoder_dict_unified_zarr",
        "FEATURES_DA3_ROOT": "/home/gl/features_model/rgb_dataset/features_da3_encoder_dict_unified_zarr",
    }

    v = os.environ.get(env, "").strip() or default_map.get(env, "")
    if not v:
        raise RuntimeError(f"请设置环境变量 {env}=/path/to/features_root")
    if not Path(v).is_dir():
        raise RuntimeError(f"{env} 不是目录: {v}")
    return v


def _assert_basic_pack(pack, *, model_name: str):
    assert isinstance(pack.meta, dict)

    # 兼容两种 pack：
    # - FeaturePack (.pt): pack.per_frame_features 是 torch.Tensor
    # - ZarrPack (.zarr):  pack.arr 是 zarr.Array（通过 pack.shape / pack.get_window 访问）
    if hasattr(pack, "per_frame_features"):
        pf = pack.per_frame_features
        assert pf is not None, f"{model_name}: per_frame_features 为空（你现在默认应该保留时间维）"
        assert torch.is_tensor(pf)
        assert pf.ndim == 5, f"{model_name}: 期望 [W,T,Hf,Wf,C]，实际 {tuple(pf.shape)}"
        w, t, hf, wf, c = pf.shape
    else:
        assert hasattr(pack, "shape"), f"{model_name}: pack 缺少 per_frame_features/shape"
        w, t, hf, wf, c = pack.shape  # type: ignore[attr-defined]
    assert w > 0 and t > 0 and hf > 0 and wf > 0 and c > 0

    # 时间维应等于 window_size（默认 8）
    win = pack.meta.get("window_size")
    if win is not None:
        assert int(win) == int(t), f"{model_name}: meta.window_size={win} 与 tensor.t={t} 不一致"

    # NaN/Inf
    # - pt: 直接检查 tensor
    # - zarr: 只抽样检查一个 window（避免把整个 episode 读入内存）
    if hasattr(pack, "per_frame_features"):
        assert not torch.isnan(pf).any().item(), f"{model_name}: per_frame_features 包含 NaN"
        assert not torch.isinf(pf).any().item(), f"{model_name}: per_frame_features 包含 Inf"
    else:
        assert hasattr(pack, "get_window"), f"{model_name}: zarr pack 缺少 get_window"
        win0 = pack.get_window(0)  # type: ignore[attr-defined]
        x = torch.from_numpy(win0)
        assert not torch.isnan(x).any().item(), f"{model_name}: per_frame_features(zarr) 包含 NaN"
        assert not torch.isinf(x).any().item(), f"{model_name}: per_frame_features(zarr) 包含 Inf"

    # frame_paths
    assert pack.frame_paths is not None, f"{model_name}: frame_paths 为空"
    assert isinstance(pack.frame_paths, list)
    assert len(pack.frame_paths) == w, f"{model_name}: len(frame_paths) != W"
    assert all(isinstance(x, list) and len(x) == t for x in pack.frame_paths), f"{model_name}: frame_paths 形状不对"


def test_schema_single_model_smoke():
    """单模型 schema 检查：用任意一个 root 也能跑起来。"""

    # 只要你设置了任意一个，就能测出基本格式问题
    any_root = (
        os.environ.get("FEATURES_CROCO_ROOT")
        or os.environ.get("FEATURES_VGGT_ROOT")
        or os.environ.get("FEATURES_DINOV3_ROOT")
        or os.environ.get("FEATURES_DA3_ROOT")
        or "/home/gl/features_model/rgb_dataset/features_croco_encoder_dict_unified_zarr"
    )

    if not Path(any_root).is_dir():
        import pytest

        pytest.skip(f"features root not found: {any_root}")

    ep = _first_episode_file(any_root)
    pack = _load_pack_any(ep)
    _assert_basic_pack(pack, model_name="any")


def test_cross_model_frame_paths_exact_match_if_same_episode():
    """如果四个 root 都提供了，并且它们都包含同一个 task/episode，则要求 frame_paths 完全一致。

    这是真正的“跨模型可对齐”硬约束：同一窗口对应同一组原始帧路径。
    """

    try:
        croco_root = _require_root("FEATURES_CROCO_ROOT")
        vggt_root = _require_root("FEATURES_VGGT_ROOT")
        dinov3_root = _require_root("FEATURES_DINOV3_ROOT")
        da3_root = _require_root("FEATURES_DA3_ROOT")
    except RuntimeError as e:
        import pytest

        pytest.skip(str(e))

    # 用 croco 的第一个 pt 作为 anchor，去另外三个 root 找同名 task/episode
    anchor_ep = _first_episode_file(croco_root)
    task = anchor_ep.parent.name
    episode = anchor_ep.stem
    suffix = anchor_ep.suffix

    others = {
        "croco": anchor_ep,
        "vggt": Path(vggt_root) / task / f"{episode}{suffix}",
        "dinov3": Path(dinov3_root) / task / f"{episode}{suffix}",
        "da3": Path(da3_root) / task / f"{episode}{suffix}",
    }

    missing = [k for k, p in others.items() if not p.exists()]
    if missing:
        import pytest

        pytest.skip(
            f"四个 root 没有同时包含同一个 task/episode（anchor={task}/{episode}）。缺少: {missing}"
        )

    packs = {k: _load_pack_any(p) for k, p in others.items()}
    for k, pack in packs.items():
        _assert_basic_pack(pack, model_name=k)

    fp0 = packs["croco"].frame_paths
    assert fp0 is not None

    for k in ["vggt", "dinov3", "da3"]:
        assert packs[k].frame_paths == fp0, f"frame_paths 不一致: {k} vs croco"
