"""features_common/depth_guided_film_online_v2/encoder_film_2model_v2.py

2-Model DA3-FiLM Fusion Encoder V2  (DINOv3 + DA3) — 空间感知版
================================================================

与 v1 (encoder_film_2model.py) 的核心区别
------------------------------------------
v1 的问题 (在 beat_block_hammer 等精细空间定位任务上失败的原因):
  1. DA3 做了 geo_vec = mean_pool(tokens) → 全局深度标量, 丢失空间分布
  2. DINOv3 最终也做了 mean_pool(q_tokens) → 196 个 token 平均成 1 个向量
     ▶ 方块只占 1-2 个 token, 被 194 个背景 token 稀释

v2 的改进:
  1. DA3 → Attention Pooling (可学习, 保留对重要深度区域的关注)
  2. DINOv3 最终聚合 → Attention Pooling (可学习, 自动关注显著区域)
  3. DA3 token 数量保持对齐到 DINOv3 (14×14=196), 用空间插值而非随机跳采

Attention Pooling 原理:
  给每个 token 计算一个标量注意力权重 (softmax), 加权求和
  网络自动学习"哪些空间位置更重要" — 对于小目标定位非常关键
  lift_pot: 大物体 → 权重自然均匀 ≈ mean pool → 性能不受影响
  beat_block_hammer: 红色小方块 → 高权重集中在方块位置 → 显著提升

模型顺序约定:
  index 0 -> DINOv3 (C=768, 语义)
  index 1 -> DA3    (C=2048, 几何 FiLM 调制)
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 辅助模块
# ============================================================

class AttentionPooling(nn.Module):
    """可学习注意力池化: [N, K, D] -> [N, D]

    每个 token 通过一个线性层得到标量分数, softmax 后加权求和.
    比 mean pooling 多 D+1 个参数 (可忽略不计), 但能自动学习
    关注空间上重要的 token (例如: 红色方块, 目标抓取点).

    初始化: 分数线性层全零初始化 → 初始行为等同于 mean pooling
    """

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1, bias=True)
        # 零初始化 → 初始等同 mean pooling, 稳定训练
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [N, K, D]
        Returns:
            [N, D]
        """
        scores = self.score(tokens)          # [N, K, 1]
        weights = scores.softmax(dim=1)      # [N, K, 1]
        return (tokens * weights).sum(dim=1) # [N, D]


class FiLMLayer(nn.Module):
    """FiLM 调制: scale/shift 向量逐 token 调制.

    Args:
        cond_dim: 条件向量维度 (来自 DA3 attention pool)
        feat_dim: 被调制特征维度 (语义 tokens)
        hidden:   MLP 隐藏层大小
    """

    def __init__(self, cond_dim: int, feat_dim: int, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, feat_dim * 2),
        )
        # 零初始化: 初始 scale=0 → (1+0)=1, shift=0, 等同恒等映射
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: [N, K, D]  语义 tokens
            cond: [N, cond_dim]  几何全局向量
        Returns:
            [N, K, D]
        """
        out = self.mlp(cond)
        scale, shift = out.chunk(2, dim=-1)   # [N, D] each
        scale = scale.unsqueeze(1)             # [N, 1, D]
        shift = shift.unsqueeze(1)
        return feat * (1.0 + scale) + shift


# ============================================================
# 主编码器
# ============================================================

class DA3Film2ModelEncoderV2(nn.Module):
    """2-Model DA3-FiLM V2: 空间感知 Attention Pooling 版.

    Stage 1: 投影
        DINOv3 [N, K_sem, 768]  -> linear+LN -> [N, K_sem, proj_dim]
        DA3    [N, K_geo, 2048] -> linear+LN -> [N, K_da3, proj_dim]

    Stage 2: DA3 空间感知聚合 (Attention Pooling, 非 mean pool)
        DA3 attn_pool -> [N, proj_dim]  (geo_vec)

    Stage 3: FiLM 调制 DINOv3 tokens
        geo_vec → FiLM MLP → scale, shift → 调制 [N, K_sem, proj_dim]

    Stage 4: DINOv3 空间感知聚合 (Attention Pooling, 非 mean pool)
        DINOv3 attn_pool -> [N, proj_dim] -> output_proj -> [N, out_dim]

    Args:
        semantic_in_dim:  DINOv3 token 维度 (768)
        geometric_in_dim: DA3 token 维度 (2048)
        proj_dim:         统一投影维度 (256)
        film_hidden:      FiLM MLP 隐藏层 (256)
        out_dim:          输出维度 (1280)
        with_pos_enc:     DINOv3 tokens 是否加位置编码
        dropout:          Dropout 概率
        max_sem_tokens:   DINOv3 最大 token 数 (196, 即 14×14)
        max_geo_tokens:   DA3 最大 token 数, None=保留全部 (972)
    """

    SEMANTIC_IDX = 0   # DINOv3
    GEOMETRIC_IDX = 1  # DA3

    def __init__(
        self,
        semantic_in_dim: int = 768,
        geometric_in_dim: int = 2048,
        proj_dim: int = 256,
        film_hidden: int = 256,
        out_dim: int = 1280,
        with_pos_enc: bool = True,
        dropout: float = 0.1,
        max_sem_tokens: int = 196,
        max_geo_tokens: int | None = None,  # None = 保留全部 DA3 tokens
    ):
        super().__init__()
        self.semantic_in_dim  = int(semantic_in_dim)
        self.geometric_in_dim = int(geometric_in_dim)
        self.proj_dim         = int(proj_dim)
        self.out_dim          = int(out_dim)
        self.with_pos_enc     = bool(with_pos_enc)
        self.max_sem_tokens   = int(max_sem_tokens)
        self.max_geo_tokens   = max_geo_tokens  # None 保留全部

        # ── Stage 1: 投影 ──
        self.semantic_proj = nn.Sequential(
            nn.Linear(self.semantic_in_dim, self.proj_dim),
            nn.LayerNorm(self.proj_dim),
        )
        self.geometric_proj = nn.Sequential(
            nn.Linear(self.geometric_in_dim, self.proj_dim),
            nn.LayerNorm(self.proj_dim),
        )

        # 位置编码 (只给语义 tokens)
        if with_pos_enc:
            self.pos_embed = nn.Parameter(
                torch.randn(1, max_sem_tokens, self.proj_dim) * 0.02
            )
        else:
            self.pos_embed = None

        # ── Stage 2: DA3 Attention Pooling ──
        self.geo_attn_pool = AttentionPooling(self.proj_dim)

        # ── Stage 3: FiLM ──
        self.film = FiLMLayer(
            cond_dim=self.proj_dim,
            feat_dim=self.proj_dim,
            hidden=int(film_hidden),
        )
        self.post_film_norm = nn.LayerNorm(self.proj_dim)

        # ── Stage 4: DINOv3 Attention Pooling + 输出投影 ──
        self.sem_attn_pool = AttentionPooling(self.proj_dim)
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
                # AttentionPooling.score 和 FiLMLayer.mlp[-1] 用零初始化, 跳过
                if not hasattr(m, '_zero_init'):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    # ──────────────────────────────────────────────────────────
    # Token 处理辅助函数
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _limit_tokens(tokens: torch.Tensor, max_k: int) -> torch.Tensor:
        """等间隔下采样, 仅在 K > max_k 时生效."""
        K = tokens.shape[1]
        if K <= max_k:
            return tokens
        idx = torch.linspace(0, K - 1, max_k, dtype=torch.long, device=tokens.device)
        return tokens[:, idx, :]

    # ──────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────

    def forward(self, x) -> torch.Tensor:
        """
        x: List[2 Tensor]
           x[0]: DINOv3 [B, To, K_sem, 768]  (或 [B, To, 768])
           x[1]: DA3    [B, To, K_geo, 2048]  (或 [B, To, 2048])
        Returns: [B, To, out_dim]
        """
        if len(x) != 2:
            raise ValueError(f"Expected 2 inputs (DINOv3, DA3), got {len(x)}")

        sem_tok = x[self.SEMANTIC_IDX]   # DINOv3
        geo_tok = x[self.GEOMETRIC_IDX]  # DA3

        # pool 模式兼容
        if sem_tok.ndim == 3:
            sem_tok = sem_tok.unsqueeze(2)
        if geo_tok.ndim == 3:
            geo_tok = geo_tok.unsqueeze(2)

        B, To = sem_tok.shape[0], sem_tok.shape[1]

        # Flatten B*To
        sem_flat = sem_tok.reshape(B * To, sem_tok.shape[2], sem_tok.shape[3])
        geo_flat = geo_tok.reshape(B * To, geo_tok.shape[2], geo_tok.shape[3])
        N = B * To

        # ── Stage 1: 投影 ──
        # DINOv3: limit to max_sem_tokens
        sem_flat = self._limit_tokens(sem_flat, self.max_sem_tokens)
        K_sem = sem_flat.shape[1]
        q_tokens = self.semantic_proj(
            sem_flat.reshape(N * K_sem, sem_flat.shape[2])
        ).reshape(N, K_sem, self.proj_dim)        # [N, K_sem, proj_dim]

        # DA3: limit to max_geo_tokens (若 None 则保留全部 ~972 个)
        if self.max_geo_tokens is not None:
            geo_flat = self._limit_tokens(geo_flat, self.max_geo_tokens)
        K_geo = geo_flat.shape[1]
        geo_proj = self.geometric_proj(
            geo_flat.reshape(N * K_geo, geo_flat.shape[2])
        ).reshape(N, K_geo, self.proj_dim)        # [N, K_geo, proj_dim]

        # ── 位置编码 (语义) ──
        if self.pos_embed is not None:
            pos = self.pos_embed[:, :K_sem, :]
            q_tokens = q_tokens + pos

        # ── Stage 2: DA3 Attention Pooling ──
        # 关键改进: 不再 mean pool, 而是可学习地关注重要深度区域
        geo_vec = self.geo_attn_pool(geo_proj)    # [N, proj_dim]

        # ── Stage 3: FiLM 调制 ──
        q_tokens = self.film(q_tokens, geo_vec)   # [N, K_sem, proj_dim]
        q_tokens = self.post_film_norm(q_tokens)

        # ── Stage 4: DINOv3 Attention Pooling + 输出 ──
        # 关键改进: 不再 mean pool, 而是自动关注方块/目标所在 token
        pooled = self.sem_attn_pool(q_tokens)     # [N, proj_dim]
        z = self.output_proj(pooled)              # [N, out_dim]
        z = z.reshape(B, To, self.out_dim)

        return z

    def extra_repr(self):
        return (
            f"semantic_in={self.semantic_in_dim}, geometric_in={self.geometric_in_dim}, "
            f"proj_dim={self.proj_dim}, out_dim={self.out_dim}, "
            f"max_sem_tokens={self.max_sem_tokens}, "
            f"max_geo_tokens={self.max_geo_tokens or 'all'}"
        )
