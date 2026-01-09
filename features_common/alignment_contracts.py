"""features_common/alignment_contracts.py

对齐训练需要的“最小契约（contract）”。

你当前的数据现实：
- 视觉：我们已经有 per-frame 的 dense patch feature map（来自四个模型）
- 点云：你有每帧 ply，但 **没有保存点↔像素 (u,v) 对应关系**

因此要做严格的点云-视觉对齐，必须补齐一种 correspondence：
- uv: 每个点在原 RGB 图像中的像素坐标（或 patch 坐标）

这个文件定义训练时的 sample 形态，后续训练脚本都按这个来。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch  # type: ignore


@dataclass
class Correspondence:
    """点↔像素/patch 对应关系。

    最小要求：
    - uv: [N,2] int32 (pixel coords in original RGB)

    可选：
    - valid: [N] bool（有些点可能投影到图像外/被裁剪）
    """

    uv: np.ndarray
    valid: Optional[np.ndarray] = None


@dataclass
class AlignmentSample:
    """对齐训练的最小样本。

    - vis_map: 视觉特征图（单帧） [Hf,Wf,D]
      （来自 adapter 后统一维度 D；也可以是原始 C，再在训练中投影）

    - pc_xyz: 点坐标 [N,3]
    - pc_feat: 点云特征 [N,D] 或者 None（如果你想端到端训练点云编码器）

    - corr: 点↔像素对应
    """

    vis_map: torch.Tensor
    pc_xyz: np.ndarray
    pc_feat: Optional[torch.Tensor]
    corr: Correspondence
    meta: dict[str, Any]
