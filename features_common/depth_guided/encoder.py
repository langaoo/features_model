"""features_common/depth_guided/encoder.py

Depth-Guided Cross-Attention Fusion Encoder
============================================

设计动机
--------
Direct Fusion（SimpleFusionEncoder）的问题：
  - 所有模型特征先 flatten 成向量，再加权求和
  - DA3 的二维空间拓扑结构被完全丢弃
  - DA3 与其他模型平等处理，其几何先验无法被特殊利用

本方案：语义-几何解耦 (Semantic-Geometric Decoupling)
  - 语义组 [CroCo, VGGT, DINOv3]：提供 Query（"我想了解哪里"）
  - 几何组 [DA3]：提供 Key/Value（"这里在空间中是什么位置"）
  - Cross-Attention：语义 tokens 通过几何 tokens 重新聚合，获得深度感知的语义特征

输入（两种模式）
---------
  pool 模式（兼容旧接口）:
    x: List[4个 Tensor，每个 [B, To, C_i]]（mean-pooled 全局向量）
  token 模式（推荐）:
    x: List[4个 Tensor，每个 [B, To, K_i, C_i]]（空间 patch tokens）

输出
----
  z: FloatTensor[B, To, out_dim]（每帧的融合表示）

模型顺序约定（与整个 DP2DP3 项目一致）
  index 0 → CroCo   (C=1024)
  index 1 → VGGT    (C=2048)
  index 2 → DINOv3  (C=768)
  index 3 → DA3     (C=2048)  ← 几何组

参数量估计（默认配置 proj_dim=512, n_heads=8, n_layers=2, out_dim=1280）
  - semantic proj (3×Linear): ~3.5M
  - geometric proj (1×Linear): ~1M
  - Cross-Attention (2层): ~8M
  - Output proj: ~1.3M
  总计 ~14M，比 tokens_full alignment encoder (82M) 小很多
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthGuidedCrossAttention(nn.Module):
    """单层 Depth-Guided Cross-Attention。

    Query 来自语义特征，Key/Value 来自几何（DA3）特征。
    标准 Multi-Head Cross-Attention，实现语义 tokens 按几何 tokens 重新聚合。

    Args:
        dim: 统一的 Q/K/V 投影维度
        n_heads: 注意力头数
        dropout: Attention dropout
    """

    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0, f"dim={dim} must be divisible by n_heads={n_heads}"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

        # Post cross-attention FFN (Pre-LN Transformer style)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q_tokens: torch.Tensor,   # [N, Ks, dim]
        kv_tokens: torch.Tensor,  # [N, Kg, dim]
    ) -> torch.Tensor:
        """
        Args:
            q_tokens:  语义 tokens, [N, Ks, dim]
            kv_tokens: 几何 tokens, [N, Kg, dim]
        Returns:
            out: [N, Ks, dim]（语义 tokens 经几何引导后更新）
        """
        N, Ks, D = q_tokens.shape
        _, Kg, _ = kv_tokens.shape

        # Pre-LN
        q_res = q_tokens
        q = self.norm_q(q_tokens)
        kv = self.norm_kv(kv_tokens)

        # Project
        Q = self.q_proj(q).reshape(N, Ks, self.n_heads, self.head_dim).transpose(1, 2)   # [N, h, Ks, hd]
        K = self.k_proj(kv).reshape(N, Kg, self.n_heads, self.head_dim).transpose(1, 2)  # [N, h, Kg, hd]
        V = self.v_proj(kv).reshape(N, Kg, self.n_heads, self.head_dim).transpose(1, 2)  # [N, h, Kg, hd]

        # Scaled dot-product attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale   # [N, h, Ks, Kg]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, V)                                  # [N, h, Ks, hd]
        out = out.transpose(1, 2).reshape(N, Ks, D)                 # [N, Ks, D]
        out = self.out_proj(out)

        # Residual connection
        out = q_res + out

        # FFN with residual
        out = out + self.ffn(self.norm_ffn(out))

        return out


class DepthGuidedFusionEncoder(nn.Module):
    """Depth-Guided Cross-Attention Fusion Encoder（完整版）。

    两阶段处理：
      Stage 1: 各模型 tokens → 各自 Linear 投影到 proj_dim
      Stage 2: 语义 tokens 融合为 Q; DA3 tokens 作为 K/V; 多层 Cross-Attention
      Stage 3: 语义 tokens mean-pool → [B, To, proj_dim] → 输出 proj → [B, To, out_dim]

    Args:
        semantic_in_dims: 语义组各模型的输入维度，顺序对应 [CroCo, VGGT, DINOv3] = (1024, 2048, 768)
        geometric_in_dim: 几何组（DA3）的输入维度 = 2048
        proj_dim: 统一的中间投影维度（Q/K/V 都在此维度做 attention）
        n_heads: Cross-Attention 头数
        n_layers: Cross-Attention 层数（堆叠）
        out_dim: 最终输出维度（与其他实验对齐，默认 1280）
        semantic_fusion: 语义组多模型合并策略
            'concat_proj': 各模型投影到 proj_dim 后 concat，再过一个 Linear 降到 proj_dim
            'mean': 各模型投影到 proj_dim 后取均值
            'weighted': 可学习加权求和
        pool: 最终 token 聚合方式 ('mean' 或 'attn')
        dropout: Dropout 概率

    模型顺序（hardcoded，与项目约定一致）：
        输入 List 的 index 0→CroCo, 1→VGGT, 2→DINOv3, 3→DA3
    """

    SEMANTIC_IDX = (0, 1, 2)   # CroCo, VGGT, DINOv3
    GEOMETRIC_IDX = 3          # DA3

    def __init__(
        self,
        semantic_in_dims: tuple[int, int, int] = (1024, 2048, 768),
        geometric_in_dim: int = 2048,
        proj_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 2,
        out_dim: int = 1280,
        semantic_fusion: str = "concat_proj",
        pool: str = "mean",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.semantic_in_dims = semantic_in_dims
        self.geometric_in_dim = geometric_in_dim
        self.proj_dim = proj_dim
        self.n_layers = n_layers
        self.out_dim = out_dim
        self.semantic_fusion = semantic_fusion
        self.pool = pool
        self.n_semantic = len(semantic_in_dims)

        # ---------- Stage 1: 各模型投影 ----------
        # 语义组：各自投影到 proj_dim
        self.semantic_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, proj_dim),
                nn.LayerNorm(proj_dim),
            )
            for d in semantic_in_dims
        ])

        # 几何组：DA3 投影到 proj_dim
        self.geometric_proj = nn.Sequential(
            nn.Linear(geometric_in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        # ---------- Stage 1b: 语义组融合 ----------
        if semantic_fusion == "concat_proj":
            # 三个语义模型 concat 后 MLP 降到 proj_dim
            self.semantic_merge = nn.Sequential(
                nn.Linear(proj_dim * self.n_semantic, proj_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(proj_dim * 2, proj_dim),
                nn.LayerNorm(proj_dim),
            )
        elif semantic_fusion == "weighted":
            self.semantic_weights = nn.Parameter(
                torch.ones(self.n_semantic) / self.n_semantic
            )
            self.semantic_merge = None
        elif semantic_fusion == "mean":
            self.semantic_merge = None
        else:
            raise ValueError(f"Unknown semantic_fusion: {semantic_fusion}")

        # ---------- Stage 2: Cross-Attention 层（可堆叠） ----------
        self.cross_attn_layers = nn.ModuleList([
            DepthGuidedCrossAttention(dim=proj_dim, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        # ---------- Stage 3: pool + 输出投影 ----------
        if pool == "attn":
            self.pool_query = nn.Parameter(torch.randn(proj_dim))
        else:
            self.pool_query = None

        self.output_proj = nn.Sequential(
            nn.Linear(proj_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化线性层。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _project_semantic(self, tokens_list: list[torch.Tensor]) -> torch.Tensor:
        """语义组 tokens 投影并融合。

        Args:
            tokens_list: 3 个 Tensor，每个 [N, K_i, C_i]
        Returns:
            sem_tokens: [N, K, proj_dim]，其中 K 取各模型 K_i 的最大值（截断对齐）
        """
        projected = []
        for i, (toks, proj) in enumerate(zip(tokens_list, self.semantic_projs)):
            # toks: [N, K_i, C_i]
            N, K_i, C_i = toks.shape
            t = proj(toks.reshape(N * K_i, C_i)).reshape(N, K_i, self.proj_dim)
            projected.append(t)

        # 统一 token 数量：取最小 K（避免 padding 引入噪声）
        min_K = min(t.shape[1] for t in projected)
        projected = [t[:, :min_K, :] for t in projected]  # [N, min_K, proj_dim] each

        if self.semantic_fusion == "concat_proj":
            # concat 后 MLP 压缩
            cat = torch.cat(projected, dim=-1)  # [N, min_K, proj_dim * n_semantic]
            N, K, _ = cat.shape
            sem = self.semantic_merge(cat.reshape(N * K, -1)).reshape(N, K, self.proj_dim)
        elif self.semantic_fusion == "weighted":
            w = torch.softmax(self.semantic_weights, dim=0)  # [n_semantic]
            stacked = torch.stack(projected, dim=0)           # [n_semantic, N, K, proj_dim]
            sem = (stacked * w.view(-1, 1, 1, 1)).sum(dim=0) # [N, K, proj_dim]
        else:  # mean
            sem = torch.stack(projected, dim=0).mean(dim=0)  # [N, K, proj_dim]

        return sem  # [N, K, proj_dim]

    def _project_geometric(self, geo_tokens: torch.Tensor) -> torch.Tensor:
        """DA3 几何 tokens 投影。

        Args:
            geo_tokens: [N, Kg, C_geo]
        Returns:
            [N, Kg, proj_dim]
        """
        N, Kg, C_geo = geo_tokens.shape
        return self.geometric_proj(
            geo_tokens.reshape(N * Kg, C_geo)
        ).reshape(N, Kg, self.proj_dim)

    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Tokens → 单向量。

        Args:
            tokens: [N, K, proj_dim]
        Returns:
            [N, proj_dim]
        """
        if self.pool == "attn" and self.pool_query is not None:
            # 学习 query 向量做加权汇聚
            scores = (tokens * self.pool_query).sum(dim=-1) / math.sqrt(self.proj_dim)
            w = torch.softmax(scores, dim=-1)              # [N, K]
            return (tokens * w.unsqueeze(-1)).sum(dim=1)  # [N, proj_dim]
        else:
            return tokens.mean(dim=1)                      # [N, proj_dim]

    # ------------------------------------------------------------------
    # Forward：支持两种输入格式
    # ------------------------------------------------------------------

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """前向推理。

        支持两种输入格式：
          1. Token 模式（完整 spatial tokens）：
             x: List[4 个 Tensor]，每个形状 [B, To, K_i, C_i]
          2. Pool 模式（兼容旧接口，mean-pooled 全局向量）：
             x: List[4 个 Tensor]，每个形状 [B, To, C_i]
             会自动 unsqueeze K 维度为 K=1

        Returns:
            z: [B, To, out_dim]
        """
        if len(x) != 4:
            raise ValueError(f"Expected 4 models, got {len(x)}")

        # 检测 token 模式 vs pool 模式
        if x[0].ndim == 3:
            # Pool 模式：[B, To, C_i] → [B, To, 1, C_i]
            x = [t.unsqueeze(2) for t in x]

        # x[i]: [B, To, K_i, C_i]
        B, To = x[0].shape[0], x[0].shape[1]

        # 拆分语义组 + 几何组
        sem_list = [x[i] for i in self.SEMANTIC_IDX]   # [B, To, K_i, C_i] × 3
        geo_tok = x[self.GEOMETRIC_IDX]                 # [B, To, Kg, C_geo]

        # ----- 合并 B, To 到 batch 维度，方便处理 -----
        # sem_list: each [B*To, K_i, C_i]
        sem_flat = [t.reshape(B * To, t.shape[2], t.shape[3]) for t in sem_list]
        geo_flat = geo_tok.reshape(B * To, geo_tok.shape[2], geo_tok.shape[3])

        # Stage 1: 投影
        q_tokens = self._project_semantic(sem_flat)  # [B*To, K, proj_dim]
        kv_tokens = self._project_geometric(geo_flat) # [B*To, Kg, proj_dim]

        # Stage 2: 多层 Cross-Attention
        out = q_tokens
        for layer in self.cross_attn_layers:
            out = layer(out, kv_tokens)              # [B*To, K, proj_dim]

        # Stage 3: pool → output proj
        pooled = self._pool_tokens(out)              # [B*To, proj_dim]
        z = self.output_proj(pooled)                 # [B*To, out_dim]
        z = z.reshape(B, To, self.out_dim)           # [B, To, out_dim]

        return z

    def extra_repr(self) -> str:
        return (
            f"semantic_in_dims={self.semantic_in_dims}, "
            f"geometric_in_dim={self.geometric_in_dim}, "
            f"proj_dim={self.proj_dim}, n_layers={self.n_layers}, "
            f"out_dim={self.out_dim}, "
            f"semantic_fusion={self.semantic_fusion}, pool={self.pool}"
        )
