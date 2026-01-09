"""features_common/adapters.py

一套“跨模型特征对齐”的 adapter 实现。

目标
- 输入：不同模型输出的 per_frame_features: [W,T,Hf,Wf,C]
- 输出：统一维度 D 的特征，支持：
  1) 仅通道对齐（C -> D）
  2) 可学习的时间聚合（T -> 1），得到 [W,Hf,Wf,D]

设计选择（稳健优先）
- ChannelAdapter：LN + Linear(C->D) (+ GELU + Linear(D->D))
- TemporalGatedPooling：对每个 patch 学一组时间权重（softmax），加权求和。

注意
- 这里不试图把不同模型的 Hf/Wf 对齐；几何对齐应在投影到 3D 时处理。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch  # type: ignore
import torch.nn as nn  # type: ignore


class ChannelAdapter(nn.Module):
    """把通道从 C 投影到统一维度 D。"""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = out_dim

        self.norm = nn.LayerNorm(in_dim)
        self.proj1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.proj2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., C] -> [..., D]"""
        x = self.norm(x)
        x = self.proj1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.proj2(x)
        return x


class TemporalGatedPooling(nn.Module):
    """对时间维 T 做可学习加权（softmax gating）。

    输入：x [W,T,Hf,Wf,D]
    输出：y [W,Hf,Wf,D]

    实现：对 token 的 D 维做一个线性打分得到 logits[T]，softmax 后加权求和。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"期望 [W,T,Hf,Wf,D]，但得到 {tuple(x.shape)}")
        # logits: [W,T,Hf,Wf,1]
        logits = self.score(x)
        w = torch.softmax(logits, dim=1)
        y = (x * w).sum(dim=1)
        return y


@dataclass
class AdapterOutput:
    per_frame: torch.Tensor  # [W,T,Hf,Wf,D]
    pooled: Optional[torch.Tensor] = None  # [W,Hf,Wf,D]


class MultiModelAdapter(nn.Module):
    """一个简单可落地的“先通道对齐，再可选时间聚合”的 adapter。

    用法：
    - 初始化时指定 out_dim=D
    - forward 时传入 per_frame_features（来自某个模型）即可

    注意：这里不强制要求输入 C 固定。
    - 如果你希望一个 adapter 同时服务多个模型，可以为每个模型单独建一个 ChannelAdapter。
    - 这里提供的是单模型版本（更清晰、也更符合你后续按模型训练/冻结的需求）。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 256,
        *,
        use_temporal_pool: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.channel = ChannelAdapter(in_dim=in_dim, out_dim=out_dim, dropout=dropout)
        self.use_temporal_pool = bool(use_temporal_pool)
        self.temporal = TemporalGatedPooling(dim=out_dim) if self.use_temporal_pool else None

    def forward(self, per_frame_features: torch.Tensor) -> AdapterOutput:
        """per_frame_features: [W,T,Hf,Wf,C]"""
        if per_frame_features.ndim != 5:
            raise ValueError(f"期望 [W,T,Hf,Wf,C]，但得到 {tuple(per_frame_features.shape)}")

        # 通道对齐：对最后一维做 MLP
        x = self.channel(per_frame_features)

        pooled = self.temporal(x) if self.temporal is not None else None
        return AdapterOutput(per_frame=x, pooled=pooled)
