"""features_common/fusion.py

多模型特征融合模块。

当前仓库已有 `tools/train_alignment_skeleton.py` 内联的 `WeightedFusion`（全局可学习权重）。
路线2（RGB-only 蒸馏）会频繁复用融合逻辑，因此抽成公共模块，并额外提供一个轻量 MoE gating：

- WeightedFusion: 全局权重（与样本无关），最稳定、最省算。
- MoEFusion: 对每个样本/点的 token 学门控权重（样本相关），表达更强但更吃算。

输入统一约定：zs = list[Tensor[N,D]]
输出：fused Tensor[N,D] 与权重（WeightedFusion: [M]；MoE: [N,M]）。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class WeightedFusion(nn.Module):
    """可学习的全局加权融合（全局权重）。"""

    def __init__(self, num_models: int):
        super().__init__()
        if num_models <= 0:
            raise ValueError("num_models must be > 0")
        self.logits = nn.Parameter(torch.zeros(num_models, dtype=torch.float32))

    def forward(self, zs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if len(zs) == 0:
            raise ValueError("zs 不能为空")
        if len(zs) == 1:
            w = torch.ones(1, device=zs[0].device, dtype=zs[0].dtype)
            return zs[0], w
        weights = torch.softmax(self.logits.to(zs[0].device), dim=0)  # [M]
        fused = 0.0
        for i, z in enumerate(zs):
            fused = fused + z * weights[i]
        return fused, weights


class MoEFusion(nn.Module):
    """Noisy Top-K MoE 门控融合。

    特性：
    1. Noisy Gating: 训练时加入噪声，促进负载均衡。
    2. Top-K: 仅保留权重最大的 K 个专家，其余置零。
    3. Softmax: 归一化权重。
    """

    def __init__(self, dim: int, num_models: int, hidden_dim: int = 256, k: int = 2, noisy: bool = True):
        super().__init__()
        if num_models <= 0:
            raise ValueError("num_models must be > 0")
        self.num_models = int(num_models)
        self.dim = int(dim)
        self.k = min(k, num_models)
        self.noisy = noisy
        
        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(self.dim * self.num_models, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_models),
        )

        # 噪声参数：per-expert 的可学习 std（log-space），用于 controllable noisy gating。
        # 初始化为较小噪声，避免训练初期过于随机。
        self.noise_log_std = nn.Parameter(torch.full((self.num_models,), -2.0, dtype=torch.float32))

    def forward(self, zs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if len(zs) == 0:
            raise ValueError("zs 不能为空")
        if len(zs) == 1:
            w = torch.ones((zs[0].shape[0], 1), device=zs[0].device, dtype=zs[0].dtype)
            return zs[0], w
        
        # [N, M*D]
        cat = torch.cat(zs, dim=-1)
        clean_logits = self.gate(cat)  # [N,M]

        if self.training and self.noisy:
            # Noisy Gating (trainable): logits + eps * std, std is per-expert and non-negative.
            # Use exp(log_std) and cast to logits dtype for AMP safety.
            std = torch.exp(self.noise_log_std).to(device=clean_logits.device, dtype=clean_logits.dtype)  # [M]
            noise = torch.randn_like(clean_logits) * std.unsqueeze(0)
            logits = clean_logits + noise
        else:
            logits = clean_logits

        # Top-K
        # indices: [N, K]
        topk_logits, topk_indices = torch.topk(logits, self.k, dim=-1)
        
        # Mask out non-top-k
        mask = torch.zeros_like(logits).scatter_(1, topk_indices, 1.0)
        
        # Softmax only on top-k (others are -inf effectively, or just masked after softmax)
        # 标准做法：对 topk_logits 做 softmax，然后 scatter 回去
        topk_weights = torch.softmax(topk_logits, dim=-1) # [N, K]
        
        # 构造完整权重矩阵 [N, M]
        w = torch.zeros_like(logits)
        w.scatter_(1, topk_indices, topk_weights.to(w.dtype))

        fused = 0.0
        for i, z in enumerate(zs):
            # w[:, i:i+1] 是第 i 个模型的权重
            fused = fused + z * w[:, i : i + 1]
            
        return fused, w
