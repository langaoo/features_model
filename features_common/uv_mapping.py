"""features_common/uv_mapping.py

把点云里的 raw 像素坐标 (u,v) 映射到不同视觉模型的 patch feature 网格坐标。

动机
----
我们的点云 PLY 里每个点都有 (u,v)，对应原始 RGB 图像像素坐标（例如 320x240）。
而四个视觉模型在提特征时会做不同的预处理（resize / crop / pad / 取整到 patch 的倍数）。
如果不严格复刻预处理，就会导致 point↔patch 对齐发生系统性偏移。

本模块的目标是：
- 只依赖“原图尺寸 + 模型预处理参数 + 输出特征网格形状(Hf,Wf)”
- 产出每个点对应的 patch index (pv,pu) 以及有效 mask

约定
----
- uv: np.ndarray [N,2]，uv[:,0]=u(列/x)，uv[:,1]=v(行/y)
- raw_hw: (H,W) 为原始 RGB 图像尺寸
- 返回的 (pv,pu) 是整数索引，满足 0<=pv<Hf, 0<=pu<Wf

实现覆盖（与仓库内提特征脚本一致）
--------------------------------
- CroCo: croco/extract_multi_frame_croco_features.py -> Resize((img_size,img_size))
- DINOv3 local: dinov3/extract_multi_frame_dinov3_features_local.py -> PIL resize((w,h))，默认 224x224
- VGGT: vggt/vggt/utils/load_fn.py -> mode='crop'(default)，宽设为518，按比例算高并 round 到14倍，若高>518则居中裁剪到518；最终输出约为 518x518
- Depth-Anything-3: Depth-Anything-3/src/depth_anything_3/utils/io/input_processor.py
  -> boundary resize + make_divisible_by_{resize|crop} (patch=14)

注意
----
1) 这里我们只做 “几何映射” （坐标变换），不做归一化等。
2) 对 DA3：我们复刻其几何规则（longest/shortest side resize + divisible-by 处理）。
   这足以把 uv 严格映射到最终输入(H,W)，再除以 14 得到 (Hf,Wf)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class UvPatchMapping:
    pv: np.ndarray  # [N] int64
    pu: np.ndarray  # [N] int64
    mask: np.ndarray  # [N] bool
    processed_hw: Tuple[int, int]  # (H,W) after preprocess


def _clip_and_mask(u: np.ndarray, v: np.ndarray, H: int, W: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把浮点像素坐标 round 到 int，并产生 in-bounds mask。"""

    ui = np.rint(u).astype(np.int64)
    vi = np.rint(v).astype(np.int64)
    mask = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui = np.clip(ui, 0, W - 1)
    vi = np.clip(vi, 0, H - 1)
    return ui, vi, mask


def _to_patch(pix_u: np.ndarray, pix_v: np.ndarray, Hf: int, Wf: int, patch: int) -> tuple[np.ndarray, np.ndarray]:
    pu = np.clip(pix_u // patch, 0, Wf - 1)
    pv = np.clip(pix_v // patch, 0, Hf - 1)
    return pv.astype(np.int64), pu.astype(np.int64)


# -------------------------
# CroCo / DINOv3：直接 resize 到正方形
# -------------------------

def map_uv_resize_square(
    uv: np.ndarray,
    raw_hw: Tuple[int, int],
    *,
    out_size: int,
    patch_size: int,
    Hf: int,
    Wf: int,
) -> UvPatchMapping:
    """raw(H,W) -> resize(out_size,out_size) -> patch grid.

    这与 CroCo 提特征脚本完全一致：Resize((img_size,img_size))。
    """

    raw_h, raw_w = int(raw_hw[0]), int(raw_hw[1])
    u = uv[:, 0].astype(np.float32)
    v = uv[:, 1].astype(np.float32)

    # scale independently
    u2 = u * (out_size / float(raw_w))
    v2 = v * (out_size / float(raw_h))

    ui, vi, mask = _clip_and_mask(u2, v2, out_size, out_size)
    pv, pu = _to_patch(ui, vi, Hf, Wf, patch_size)
    return UvPatchMapping(pv=pv, pu=pu, mask=mask, processed_hw=(out_size, out_size))


# -------------------------
# VGGT：宽到518 + 高按比例 + round到14倍 + (可选)居中裁剪高到518
# -------------------------

def map_uv_vggt_crop_mode(
    uv: np.ndarray,
    raw_hw: Tuple[int, int],
    *,
    target_size: int = 518,
    patch_size: int = 14,
    Hf: Optional[int] = None,
    Wf: Optional[int] = None,
) -> UvPatchMapping:
    """复刻 vggt/vggt/utils/load_fn.py 中 mode='crop' 的几何。

    步骤：
    1) new_width = target_size
    2) new_height = round( raw_h * (new_width/raw_w) / 14 ) * 14
    3) resize 到 (new_width, new_height)
    4) 若 new_height > target_size：center crop 到高度 target_size（宽不变）

    最终 processed 尺寸通常是 (target_size, target_size)。
    """

    raw_h, raw_w = int(raw_hw[0]), int(raw_hw[1])
    new_w = int(target_size)
    new_h = int(round((raw_h * (new_w / float(raw_w))) / patch_size) * patch_size)

    u = uv[:, 0].astype(np.float32)
    v = uv[:, 1].astype(np.float32)

    # resize坐标
    u2 = u * (new_w / float(raw_w))
    v2 = v * (new_h / float(raw_h))

    # crop（只裁高度）
    crop_top = 0
    out_h = new_h
    out_w = new_w
    if new_h > target_size:
        crop_top = (new_h - target_size) // 2
        out_h = target_size
        v2 = v2 - float(crop_top)

    ui, vi, mask = _clip_and_mask(u2, v2, out_h, out_w)

    if Hf is None:
        Hf = out_h // patch_size
    if Wf is None:
        Wf = out_w // patch_size

    pv, pu = _to_patch(ui, vi, int(Hf), int(Wf), patch_size)
    return UvPatchMapping(pv=pv, pu=pu, mask=mask, processed_hw=(out_h, out_w))


# -------------------------
# DA3：boundary resize + divisible-by resize/crop（patch=14）
# -------------------------

def _round_to_nearest_multiple(x: float, m: int) -> int:
    return int(np.round(x / m) * m)


def _floor_to_multiple(x: float, m: int) -> int:
    return int(np.floor(x / m) * m)


def map_uv_da3(
    uv: np.ndarray,
    raw_hw: Tuple[int, int],
    *,
    process_res: int,
    process_res_method: Literal[
        "upper_bound_resize",
        "lower_bound_resize",
        "upper_bound_crop",
        "lower_bound_crop",
    ] = "upper_bound_resize",
    patch_size: int = 14,
    Hf: Optional[int] = None,
    Wf: Optional[int] = None,
) -> UvPatchMapping:
    """复刻 DA3 InputProcessor 的几何部分。

    1) boundary resize：保持长宽比
       - upper_bound_*: 缩放使 longest_side == process_res
       - lower_bound_*: 缩放使 shortest_side == process_res
    2) divisible-by PATCH_SIZE：
       - *resize: 每个维度 round 到最近的 multiple（会轻微缩放）
       - *crop:  每个维度 floor 到 multiple，然后做 center crop

    备注：DA3 实现里按 PIL size 顺序是 (W,H)。这里我们统一用 (H,W)。
    """

    raw_h, raw_w = int(raw_hw[0]), int(raw_hw[1])
    u = uv[:, 0].astype(np.float32)
    v = uv[:, 1].astype(np.float32)

    # --- boundary resize ---
    longest = max(raw_w, raw_h)
    shortest = min(raw_w, raw_h)
    if process_res_method.startswith("upper_bound"):
        scale = process_res / float(longest)
    else:
        scale = process_res / float(shortest)

    w1 = raw_w * scale
    h1 = raw_h * scale
    # DA3 uses cv2.resize -> int round; we'll mimic by round to int
    w1_i = int(np.round(w1))
    h1_i = int(np.round(h1))

    u1 = u * scale
    v1 = v * scale

    # --- make divisible ---
    if process_res_method.endswith("resize"):
        # round each dim to nearest multiple via small resize
        w2 = _round_to_nearest_multiple(w1_i, patch_size)
        h2 = _round_to_nearest_multiple(h1_i, patch_size)
        # additional scale per axis
        sx = w2 / float(w1_i) if w1_i > 0 else 1.0
        sy = h2 / float(h1_i) if h1_i > 0 else 1.0
        u2 = u1 * sx
        v2 = v1 * sy
        out_w, out_h = int(w2), int(h2)
    else:
        # floor each dim to multiple via center crop
        w2 = _floor_to_multiple(w1_i, patch_size)
        h2 = _floor_to_multiple(h1_i, patch_size)
        crop_left = max(0, (w1_i - w2) // 2)
        crop_top = max(0, (h1_i - h2) // 2)
        u2 = u1 - float(crop_left)
        v2 = v1 - float(crop_top)
        out_w, out_h = int(w2), int(h2)

    ui, vi, mask = _clip_and_mask(u2, v2, out_h, out_w)

    if Hf is None:
        Hf = out_h // patch_size
    if Wf is None:
        Wf = out_w // patch_size

    pv, pu = _to_patch(ui, vi, int(Hf), int(Wf), patch_size)
    return UvPatchMapping(pv=pv, pu=pu, mask=mask, processed_hw=(out_h, out_w))
