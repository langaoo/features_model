"""features_common/depth_guided_film_online/encoder_film_2model.py

2-Model DA3-FiLM Fusion Encoder  (DINOv3 + DA3)
================================================

简化自 depth_guided/encoder_film.py (4 模型版, 96%):
  - 去掉 CroCo / VGGT, 只保留 DINOv3 作为语义来源
  - DA3 作为几何调制信号 (FiLM)
  - 无需 concat_proj 融合 (只有 1 个语义模型)

模型顺序约定:
  index 0 -> DINOv3 (C=768, 语义)
  index 1 -> DA3    (C=2048, 几何 FiLM 调制)

核心公式:
  sem_tokens = DINOv3_proj(tokens)            # [B*To, K, proj_dim]
  geo_vec    = mean_pool(DA3_proj(tokens))    # [B*To, proj_dim]
  scale, shift = film_mlp(geo_vec)            # [B*To, proj_dim] each
  fused      = sem_tokens * (1 + scale) + shift
  output     = output_proj(mean_pool(fused))  # [B, To, out_dim]
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMLayer(nn.Module):
    """FiLM 调制层: scale/shift 向量逐 token 调制.

    Args:
        cond_dim:  条件向量维度 (来自 DA3 几何 mean_pool)
        feat_dim:  被调制特征维度 (语义 tokens)
        hidden:    MLP 隐藏层大小
    """
    def __init__(self, cond_dim: int, feat_dim: int, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, feat_dim * 2),  # -> [scale, shift]
        )
        # 零初始化: 训练初期 scale=0 -> 1+0=1, shift=0, 等同恒等映射
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: [N, K, D]  语义 tokens
            cond: [N, cond_dim]  几何全局向量
        Returns:
            [N, K, D]  调制后特征
        """
        out = self.mlp(cond)                  # [N, D*2]
        scale, shift = out.chunk(2, dim=-1)   # [N, D] each
        scale = scale.unsqueeze(1)            # [N, 1, D]
        shift = shift.unsqueeze(1)
        return feat * (1.0 + scale) + shift


class DA3Film2ModelEncoder(nn.Module):
    """2-Model DA3-FiLM Fusion Encoder (DINOv3 语义 + DA3 几何调制).

    三阶段处理:
      Stage 1: DINOv3 tokens -> Linear+LN -> proj_dim
               DA3 tokens    -> Linear+LN -> proj_dim
      Stage 2: DA3 mean_pool -> FiLM MLP -> scale/shift -> 调制 DINOv3 tokens
      Stage 3: 调制后 tokens mean_pool -> output_proj -> out_dim

    Args:
        semantic_in_dim:  DINOv3 token 维度 (768)
        geometric_in_dim: DA3 token 维度 (2048)
        proj_dim:         统一投影维度 (256)
        film_hidden:      FiLM MLP 隐藏层大小 (256)
        out_dim:          最终输出维度 (1280, 与其他实验对齐)
        with_pos_enc:     是否对语义 tokens 加 1D learnable 位置编码
        dropout:          Dropout 概率
        max_tokens:       每模型最大 token 数 (等间隔下采样), 默认 196
    """

    SEMANTIC_IDX = 0     # DINOv3
    GEOMETRIC_IDX = 1    # DA3

    def __init__(
        self,
        semantic_in_dim: int = 768,
        geometric_in_dim: int = 2048,
        proj_dim: int = 256,
        film_hidden: int = 256,
        out_dim: int = 1280,
        with_pos_enc: bool = True,
        dropout: float = 0.1,
        max_tokens: int = 196,
    ):
        super().__init__()
        self.semantic_in_dim = int(semantic_in_dim)
        self.geometric_in_dim = int(geometric_in_dim)
        self.proj_dim = int(proj_dim)
        self.out_dim = int(out_dim)
        self.with_pos_enc = bool(with_pos_enc)
        self.max_tokens = int(max_tokens)

        # Stage 1: 独立投影
        self.semantic_proj = nn.Sequential(
            nn.Linear(self.semantic_in_dim, self.proj_dim),
            nn.LayerNorm(self.proj_dim),
        )
        self.geometric_proj = nn.Sequential(
            nn.Linear(self.geometric_in_dim, self.proj_dim),
            nn.LayerNorm(self.proj_dim),
        )

        # 可选 1D learnable 位置编码
        if with_pos_enc:
            self.pos_embed = nn.Parameter(
                torch.randn(1, max_tokens, self.proj_dim) * 0.02
            )
        else:
            self.pos_embed = None

        # Stage 2: DA3-FiLM 调制
        self.film = FiLMLayer(
            cond_dim=self.proj_dim,
            feat_dim=self.proj_dim,
            hidden=int(film_hidden),
        )
        self.post_film_norm = nn.LayerNorm(self.proj_dim)

        # Stage 3: pool + 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(self.proj_dim, self.out_dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.out_dim * 2, self.out_dim),
            nn.LayerNorm(self.out_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _sample_tokens(self, tokens: torch.Tensor, max_k: int) -> torch.Tensor:
        """等间隔下采样到 max_k."""
        K = tokens.shape[1]
        if K <= max_k:
            return tokens
        idx = torch.linspace(0, K - 1, max_k, dtype=torch.long, device=tokens.device)
        return tokens[:, idx, :]

    def forward(self, x) -> torch.Tensor:
        """
        x: List[2 个 Tensor]
           - x[0]: DINOv3, [B, To, K_sem, 768]  (token 模式)
                    或 [B, To, 768] (pool 模式, 自动 unsqueeze)
           - x[1]: DA3,    [B, To, K_geo, 2048]
                    或 [B, To, 2048]
        Returns: [B, To, out_dim]
        """
        if len(x) != 2:
            raise ValueError(f"Expected 2 models (DINOv3, DA3), got {len(x)}")

        sem_tok = x[self.SEMANTIC_IDX]
        geo_tok = x[self.GEOMETRIC_IDX]

        # Pool 模式兼容: [B, To, C] -> [B, To, 1, C]
        if sem_tok.ndim == 3:
            sem_tok = sem_tok.unsqueeze(2)
        if geo_tok.ndim == 3:
            geo_tok = geo_tok.unsqueeze(2)

        B, To = sem_tok.shape[0], sem_tok.shape[1]

        # Flatten B, To -> batch 维度
        sem_flat = sem_tok.reshape(B * To, sem_tok.shape[2], sem_tok.shape[3])  # [N, K_sem, 768]
        geo_flat = geo_tok.reshape(B * To, geo_tok.shape[2], geo_tok.shape[3])  # [N, K_geo, 2048]

        # Stage 1: 投影
        N_s, K_s, C_s = sem_flat.shape
        q_tokens = self.semantic_proj(
            sem_flat.reshape(N_s * K_s, C_s)
        ).reshape(N_s, K_s, self.proj_dim)
        q_tokens = self._sample_tokens(q_tokens, self.max_tokens)
        K = q_tokens.shape[1]

        N_g, K_g, C_g = geo_flat.shape
        geo_proj = self.geometric_proj(
            geo_flat.reshape(N_g * K_g, C_g)
        ).reshape(N_g, K_g, self.proj_dim)
        geo_proj = self._sample_tokens(geo_proj, self.max_tokens)
        geo_vec = geo_proj.mean(dim=1)  # [N, proj_dim]

        # 位置编码
        if self.pos_embed is not None:
            pos = self.pos_embed[:, :K, :]
            q_tokens = q_tokens + pos

        # Stage 2: DA3-FiLM 调制
        q_tokens = self.film(q_tokens, geo_vec)
        q_tokens = self.post_film_norm(q_tokens)

        # Stage 3: mean_pool -> output_proj
        pooled = q_tokens.mean(dim=1)          # [N, proj_dim]
        z = self.output_proj(pooled)           # [N, out_dim]
        z = z.reshape(B, To, self.out_dim)

        return z

    def extra_repr(self):
        return (
            f"semantic_in_dim={self.semantic_in_dim}, "
            f"geometric_in_dim={self.geometric_in_dim}, "
            f"proj_dim={self.proj_dim}, out_dim={self.out_dim}, "
            f"with_pos_enc={self.with_pos_enc}, max_tokens={self.max_tokens}"
        )
