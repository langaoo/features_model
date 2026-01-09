# -*- coding: utf-8 -*-
"""uv->patch 映射的单元测试。

这些测试不依赖真实模型，只验证几何映射的边界与一致性：
- 输出索引不越界
- mask 合理（落在图内的点应尽量有效）

注意：这里只覆盖核心规则（CroCo/DINOv3 的 square resize、VGGT crop-mode、DA3 的 boundary+divisible）。
"""

from __future__ import annotations

import numpy as np

from features_common.uv_mapping import (
    map_uv_da3,
    map_uv_resize_square,
    map_uv_vggt_crop_mode,
)


def _corners_uv(w: int, h: int) -> np.ndarray:
    # (u,v) 像素坐标：包含四角 + 中心
    return np.asarray(
        [
            [0.0, 0.0],
            [w - 1.0, 0.0],
            [0.0, h - 1.0],
            [w - 1.0, h - 1.0],
            [(w - 1.0) / 2.0, (h - 1.0) / 2.0],
        ],
        dtype=np.float32,
    )


def test_map_uv_resize_square_in_bounds():
    raw_h, raw_w = 240, 320
    out_size = 224
    patch = 14
    hf = out_size // patch
    wf = out_size // patch

    uv = _corners_uv(raw_w, raw_h)
    m = map_uv_resize_square(
        uv=uv,
        raw_hw=(raw_h, raw_w),
        out_size=out_size,
        patch_size=patch,
        Hf=hf,
        Wf=wf,
    )

    assert m.pu.shape == (uv.shape[0],)
    assert m.pv.shape == (uv.shape[0],)
    assert m.mask.shape == (uv.shape[0],)

    assert m.processed_hw == (out_size, out_size)
    assert np.all((m.pu >= 0) & (m.pu < wf))
    assert np.all((m.pv >= 0) & (m.pv < hf))
    assert bool(m.mask.all())


def test_map_uv_vggt_crop_mode_basic():
    # raw 320x240 -> width fixed 518, height ~389 (round14), no crop
    raw_h, raw_w = 240, 320
    patch = 14

    uv = _corners_uv(raw_w, raw_h)
    m = map_uv_vggt_crop_mode(
        uv=uv,
        raw_hw=(raw_h, raw_w),
        target_size=518,
        patch_size=patch,
    )

    gh, gw = m.processed_hw[0] // patch, m.processed_hw[1] // patch
    assert gw == 518 // patch
    assert gh > 0

    assert np.all((m.pu >= 0) & (m.pu < gw))
    assert np.all((m.pv >= 0) & (m.pv < gh))
    assert bool(m.mask.all())


def test_map_uv_da3_in_bounds_upper_bound_resize():
    # 用一个常见 case：raw 320x240，经 upper_bound_resize 到最长边=518
    raw_h, raw_w = 240, 320
    patch = 14

    uv = _corners_uv(raw_w, raw_h)
    m = map_uv_da3(
        uv=uv,
        raw_hw=(raw_h, raw_w),
        process_res=518,
        process_res_method="upper_bound_resize",
        patch_size=patch,
    )

    gh, gw = m.processed_hw[0] // patch, m.processed_hw[1] // patch
    assert gh > 0 and gw > 0
    assert np.all((m.pu >= 0) & (m.pu < gw))
    assert np.all((m.pv >= 0) & (m.pv < gh))
    assert bool(m.mask.all())
